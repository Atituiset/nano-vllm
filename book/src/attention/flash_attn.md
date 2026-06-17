# 5.1 Flash Attention 调用

## Attention 类 (`attention.py:43-75`)

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v, ...)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, ...)
        return o
```

## 三条执行路径

### 1. Prefill（无前缀缓存）

```python
o = flash_attn_varlen_func(q, k, v,
    max_seqlen_q=context.max_seqlen_q,
    cu_seqlens_q=context.cu_seqlens_q,
    max_seqlen_k=context.max_seqlen_k,
    cu_seqlens_k=context.cu_seqlens_k,
    softmax_scale=self.scale, causal=True)
```

- Q 和 K/V 都来自当前计算的 k/v 张量
- 使用变长序列格式（`cu_seqlens` 描述每个序列的边界）
- 因果注意力：每个 token 只能看到自己及之前的 token

### 2. Prefill（有前缀缓存）

```python
if context.block_tables is not None:
    k, v = k_cache, v_cache
o = flash_attn_varlen_func(q, k, v, ...,
    block_table=context.block_tables)
```

- K/V 直接使用 `k_cache, v_cache`——FlashAttention 内部通过 `block_table` 从物理块中读取
- 当前步新计算的 K/V 已经通过 `store_kvcache` 写入了 cache
- `cu_seqlens_k` 比 `cu_seqlens_q` 大——因为它包含了缓存的历史 token

### 3. Decode

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),    # [bs, 1, heads, dim] → FlashAttention decode 格式
    k_cache, v_cache,
    cache_seqlens=context.context_lens,   # 每个序列的上下文长度
    block_table=context.block_tables,
    softmax_scale=self.scale, causal=True)
```

- Q 只有 1 个 token per 序列
- K/V 完全从 cache 中读取
- `cache_seqlens` 告诉 FlashAttention 每个序列在 cache 中有多少有效 token
- `block_table` 用于从物理块中定位每个序列的 K/V

## GQA 支持

Qwen3 使用 Grouped Query Attention（GQA）：4 个 Q 头共享 1 个 KV 头。

FlashAttention 原生支持 GQA——Q 头数 > KV 头数时，自动做 KV 头的广播/重复。
