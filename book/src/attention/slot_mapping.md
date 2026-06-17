# 5.3 Slot Mapping 计算详解

Slot Mapping 是连接 BlockManager 逻辑视图和 KV Cache 物理存储的桥梁。

## 什么是 Slot？

KV Cache 张量的布局（每层）：

```
k_cache: [num_blocks × block_size, num_kv_heads × head_dim]
```

一个 **slot** 就是 `num_blocks × block_size` 维度上的一个索引——它唯一标识了 KV Cache 中一个 token 的存储位置。

```
slot = physical_block_id × block_size + offset_in_block
```

## Prefill 阶段的 Slot Mapping (`model_runner.py:151-161`)

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size       # 起始块内偏移
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size  # 完整块
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size  # 最后块
    slot_mapping.extend(range(slot_start, slot_end))
```

### 具体例子

假设 `block_size = 4`，`block_table = [3, 7, 5]`，`start = 2, end = 9`：

```
逻辑块 0 (block_id=3): 位置 0-3    start=2 跳过 0,1
  → slot 3×4+2=14, 3×4+3=15

逻辑块 1 (block_id=7): 位置 4-7    完整块
  → slot 7×4+0=28, 7×4+1=29, 7×4+2=30, 7×4+3=31

逻辑块 2 (block_id=5): 位置 8-9    end=9 到 9-2×4=1
  → slot 5×4+0=20, 5×4+1=21

slot_mapping = [14, 15, 28, 29, 30, 31, 20, 21]
```

**注意**：slot 的值不一定递增——因为物理块在显存上不连续。这就是 PagedAttention 需要间接映射的根本原因。

## Decode 阶段的 Slot Mapping (`model_runner.py:181`)

```python
slot_mapping.append(
    seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
)
```

Decode 每步只写 1 个 token，它位于序列最后一个逻辑块的最后位置。

## Slot Mapping 的消费方

Triton `store_kvcache_kernel` 中的关键代码：

```python
slot = tl.load(slot_mapping_ptr + idx)
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
```

`slot * D` 直接索引到 KV Cache 中该 token 的起始位置。

## 与 vLLM 的对比

| | nano-vLLM | vLLM |
|--|-----------|------|
| Slot 计算 | Python 循环，CPU 端 | CUDA kernel（`reshape_and_cache` 的 `slot_mapping` 参数） |
| 数据传输 | host→device 拷贝 | 直接在 GPU 端计算 |
| 性能影响 | 长序列时 CPU 计算可能有瓶颈 | GPU 端计算无瓶颈 |

nano 的实现在功能上等价，但 CPU 端计算 slot mapping 在超长序列时可能成为性能瓶颈——vLLM 将这个计算也搬到了 GPU 上。
