# 2.4 引用计数与块回收

## 引用计数机制

Block 的 `ref_count` 是核心——它决定了物理块何时能被回收。

### 增加引用

1. **`allocate` 时复用已缓存块**：`block.ref_count += 1`（`block_manager.py:84`）
2. **`allocate` 时从 free 池取出**：`block.ref_count = 1`（`block_manager.py:86`）
3. **`may_append` 分配新块**：通过 `_allocate_block` → `block.reset()` → `ref_count = 1`

### 减少引用

唯一入口是 `deallocate`（`block_manager.py:97`）：

```python
block.ref_count -= 1
if block.ref_count == 0:
    self._deallocate_block(block_id)
```

### 引用计数变化图

```
时间    请求A block[3]  请求B block[3]   ref_count
t0      allocate        -                 1
t1      -               allocate(缓存命中) 2
t2      finish+deallocate -              1
t3      -               finish+deallocate 0 → 回收
```

## `_allocate_block` 的微妙之处 (`block_manager.py:43-51`)

```python
def _allocate_block(self):
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id
```

分配一个物理块时：

1. 从 `free_block_ids` 队头取出（FIFO）
2. 断言 `ref_count == 0`
3. **删除该块的哈希映射**：如果这个块之前有哈希（被回收前是缓存块），需要清除 `hash_to_block_id` 中的条目。否则新请求可能错误地命中已被覆盖的块
4. 重置块状态（`ref_count=1, hash=-1, token_ids=[]`）
5. 加入 `used_block_ids`

## `_deallocate_block` (`block_manager.py:53-56`)

```python
def _deallocate_block(self, block_id):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

回收时追加到 `free_block_ids` 队尾——**FIFO 回收策略**。

注意：回收时**不清除** `block.hash` 和 `hash_to_block_id`！这允许刚释放的块仍可被前缀缓存命中。直到该块被 `_allocate_block` 重新分配时才清除哈希映射。

## 潜在问题与 vLLM 的差异

nano-vLLM 的哈希条目在**物理块被重新分配**时才清除。如果大量短序列频繁分配/释放，可能出现：

- `hash_to_block_id` 中指向的 `block_id` 虽然在 `free_block_ids` 中但 `ref_count == 0`，它在 `_allocate_block` 时才会被清除
- vLLM 通过 `BlockAllocator` 的 `evictor` 更精细地管理缓存淘汰策略（LRU）

nano 的简洁实现选择了**能命中就命中，不能命中就算了**的务实策略。
