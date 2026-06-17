# 5.4 Prefill vs Decode 路径

## 完整路径对比

```
Prefill 路径:
  input_ids (变长展平)
      │
      ▼
  Embedding
      │
      ▼
  Transformer Layers × N
      │  ├─ QKV Projection
      │  ├─ RoPE
      │  ├─ store_kvcache()      ← 写入新 token 的 K/V
      │  ├─ flash_attn_varlen_func()  ← 变长注意力
      │  └─ MLP
      │
      ▼
  取每个序列最后一个 token 的 hidden state
      │
      ▼
  LM Head → logits → sample 1 token per sequence

Decode 路径:
  input_ids (每序列 1 token)
      │
      ▼
  Embedding
      │
      ▼
  Transformer Layers × N  (CUDA Graph 重放)
      │  ├─ QKV Projection
      │  ├─ RoPE
      │  ├─ store_kvcache()      ← 写入新 token 的 K/V
      │  ├─ flash_attn_with_kvcache()  ← 从 cache 读取全部 K/V
      │  └─ MLP
      │
      ▼
  LM Head → logits → sample 1 token per sequence
```

## 核心差异

### 1. FlashAttention 函数选择

| | Prefill | Decode |
|--|---------|--------|
| 函数 | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| 输入格式 | 展平所有序列的 token + `cu_seqlens` | 每序列 [1, heads, dim] + `context_lens` + `block_table` |
| K/V 来源 | 新计算的 k/v（或+cache） | 纯从 cache 读取 |
| 计算复杂度 | O(sum_len²) | O(sum_len) |

### 2. LM Head 只取最后一个 token

Prefill 时，`ParallelLMHead.forward` 只取每个序列的最后一个 token：

```python
# embed_head.py:58-60
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

这是因为只有最后一个 token 的 logits 需要被采样。

Decode 时，每个序列本来只有 1 个 token，无需裁剪。

### 3. store_kvcache 的调用

**两者都调用**——无论是 prefill 还是 decode，新计算的 K/V 都需要写入 cache。

区别在于：
- Prefill：可能写入多个 token 的 K/V（整个 prompt 或 chunk）
- Decode：只写 1 个 token 的 K/V

### 4. CUDA Graph 的使用

| | Prefill | Decode |
|--|---------|--------|
| CUDA Graph | ❌（序列长度变化太大） | ✅（固定的 batch_size 桶） |
| 执行方式 | 直接 `model()` | `graph.replay()` |
