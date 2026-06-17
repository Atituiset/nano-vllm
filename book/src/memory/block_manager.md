# 2.2 BlockManager 详解

BlockManager 是 nano-vLLM 的"内存分配器"——管理所有物理块的分配、回收和共享。

## 数据结构 (`block_manager.py:26-33`)

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()    # 哈希 → 物理块 ID
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids: set[int] = set()              # 已用块集合
```

### Block (`block_manager.py:8-24`)

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0    # 引用计数：有多少序列共享此块
        self.hash = -1        # 内容哈希（用于前缀缓存）
        self.token_ids = []   # 块内 token id（用于前缀缓存验证）
```

## 四个核心操作

### 1. `can_allocate(seq)` — 检查是否能为序列分配块 (`block_manager.py:58-73`)

```python
def can_allocate(self, seq):
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

**逻辑**：

1. 逐块计算序列的哈希链
2. 在 `hash_to_block_id` 中查找：是否已有内容相同且哈希匹配的块？
3. 若命中，该块可复用（`num_cached_blocks += 1`）
4. 若命中的块已在 `used_block_ids` 中（被其他序列持有），无需额外分配；否则需要从 free 池取出
5. 最终检查空闲块是否够分配 `num_new_blocks` 个新块
6. 返回 `num_cached_blocks`（-1 表示空间不足）

**为什么不检查最后一个块？** 因为最后一个块可能尚未填满（不满 256 token），内容不完整，哈希不具备确定性。

### 2. `allocate(seq, num_cached_blocks)` — 为序列建立 block_table (`block_manager.py:75-92`)

```python
def allocate(self, seq, num_cached_blocks):
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1      # 共享已有块
        else:
            block.ref_count = 1       # 从 free 池取出
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())  # 分配新块
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

**关键**：已缓存的前缀块通过增加引用计数共享，不占用新的物理块。

### 3. `deallocate(seq)` — 释放序列的全部块 (`block_manager.py:94-101`)

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**逆序释放**：从最后一个块开始释放，这与操作系统释放页表类似。

引用计数归零时才真正回收物理块——确保共享该块的其他序列不受影响。

### 4. `may_append(seq)` — Decode 阶段追加新块 (`block_manager.py:106-108`)

```python
def may_append(self, seq):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

**为什么是 `== 1`？** Decode 每步增加 1 token。当序列长度从 `k × block_size` 变为 `k × block_size + 1` 时，当前最后一个块刚好填满，新 token 需要一个新块。

`can_append(seq)` (`block_manager.py:103-104`) 检查条件 `len(free_block_ids) >= (len(seq) % block_size == 1)`，即仅在需要新块时才检查空闲块数量是否够。
