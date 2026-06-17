# 5.2 Triton KV Cache 写入核

KV Cache 写入是 PagedAttention 的关键操作——将新计算的 K/V 张量写入物理块对应的位置。

## 问题：为什么不直接赋值？

KV Cache 的物理块在显存上**不连续**。一个序列的 KV Cache 可能分散在物理块 3、7、12 中。简单的连续写入行不通。

需要一个"scatter write"操作：对于每个 token，计算它应该写入 KV Cache 的哪个位置（slot），然后执行写入。

## Triton Kernel (`attention.py:10-30`)

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)                 # 每个 token 一个 program
    slot = tl.load(slot_mapping_ptr + idx)  # 读取该 token 的 slot
    if slot == -1: return                   # -1 = 无需写入（CUDA Graph 填充）
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)   # 读取 K 向量
    value = tl.load(value_ptr + value_offsets)  # 读取 V 向量
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)     # 写入 K cache
    tl.store(v_cache_ptr + cache_offsets, value)   # 写入 V cache
```

## 调用方式 (`attention.py:33-40`)

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim                # 所有头展平的维度
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                               k_cache, v_cache, slot_mapping, D)
```

- `N` = 总 token 数 = 启动的 program 数
- `D = num_heads × head_dim`，作为编译时常量（`tl.constexpr`）
- 每个 program 处理 1 个 token 的 K 和 V

## 关键约束（assert 检查）

```python
assert key.stride(-1) == 1 and value.stride(-1) == 1          # 最内维连续
assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 头维度步长
assert k_cache.stride(1) == D and v_cache.stride(1) == D      # cache 展平头维度
```

Cache 的存储格式是 `[num_blocks × block_size, num_heads × head_dim]`——每个 token 的所有头连续存储。这与 `kv_cache` 张量的 `[layer, block, token, head, dim]` 是一致的，因为 PyTorch 保证最后几个维度的 stride 是连续的。

## slot=-1 的妙用

CUDA Graph 重放时，`slot_mapping` 中实际序列长度以外的位置被填充为 `-1`。kernel 检测到 `-1` 直接 `return`，避免写入无效位置。这使得不同 batch size 可以复用同一个捕获的图。
