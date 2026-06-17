# 2.3 前缀缓存机制

前缀缓存（Prefix Caching）是 PagedAttention 最重要的优化之一：当多个请求共享相同的 prompt 前缀时（如相同的 system prompt），它们可以**复用同一组物理块**，节省显存和计算。

## 哈希链设计

nano-vLLM 采用**链式哈希**：每个块的哈希值依赖于前序所有块的哈希。

`block_manager.py:36-41`：

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

```
block[0].hash = xxhash(token_ids[0])
block[1].hash = xxhash(block[0].hash || token_ids[1])
block[2].hash = xxhash(block[1].hash || token_ids[2])
...
```

这种设计保证了：

1. **前缀一致性**：相同 token 前缀必然产生相同的哈希链
2. **O(1) 查找**：通过 `hash_to_block_id` 直接定位物理块
3. **内容验证**：即使哈希碰撞，`can_allocate` 中还验证了 `block.token_ids != token_ids`

## 举个例子

假设两条请求：

```
请求A: [system_prompt...] + "请翻译这段话"
请求B: [system_prompt...] + "请总结这段话"
```

其中 `[system_prompt...]` 占 512 token = 2 个块。

```
时间线：
t1: 请求A allocate → 哈希计算 block0, block1 → 分配物理块 3, 7
    → hash_to_block_id[hash0] = 3, hash_to_block_id[hash1] = 7
    → blocks[3].ref_count = 1, blocks[7].ref_count = 1

t2: 请求B allocate → 哈希计算 block0, block1 → 命中！
    → 直接复用物理块 3, 7
    → blocks[3].ref_count = 2, blocks[7].ref_count = 2
    → 只需为 block2 以后分配新物理块
```

## 哈希更新时机

`hash_blocks` 在 `scheduler.postprocess()` 中每步调用 (`block_manager.py:110-120`)：

```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

只在某个块**刚被填满**时（`start != end`）才更新哈希。未满的最后一个块不参与哈希——因为后续 token 可能改变其内容。

## 为什么不用内容直接做 key？

`hash_to_block_id` 用哈希而非 `token_ids` 元组做 key，因为：

1. 哈希是固定 64bit，序列化开销极小
2. token_ids 列表可变且长度不固定，不适合做 dict key
3. 通过 `block.token_ids != token_ids` 验证，碰撞概率可忽略
