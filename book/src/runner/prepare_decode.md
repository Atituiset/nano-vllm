# 4.4 Decode 数据准备

Decode 阶段每个序列只生成 1 个 token，数据准备比 prefill 简单得多。

## `prepare_decode` (`model_runner.py:172-188`)

```python
def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)                    # 上一步生成的 token
        positions.append(len(seq) - 1)                      # 当前位置
        context_lens.append(len(seq))                       # 上下文长度
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )
    # ... 转移到 GPU
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

### Slot Mapping 计算

```python
slot = block_table[-1] * block_size + last_block_num_tokens - 1
```

Decode 每步只写 1 个 token 的 K/V，它属于序列的最后一个块：

- `block_table[-1]`：最后一个块的物理块 ID
- `last_block_num_tokens - 1`：该块内当前 token 的偏移（-1 因为 0-indexed）

### Context Lengths

`context_lens` 告诉 `flash_attn_with_kvcache` 每个序列在 KV Cache 中有多少有效 token——即 Q 需要关注的历史长度。

### Block Tables

Decode 阶段**总是需要** block_tables——因为 FlashAttention 需要从 KV Cache 的物理块中读取所有历史 K/V。

## Prefill vs Decode 数据对比

| | Prefill | Decode |
|--|---------|--------|
| 输入 token 数 | 多个（整个 prompt 或 chunk） | 1 |
| FlashAttention 函数 | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| cu_seqlens | ✅ | ❌ |
| context_lens | ❌ | ✅ |
| block_tables | 仅前缀缓存时 | ✅ 总是 |
| slot_mapping | 一段连续 slot | 单个 slot |
| Q 形状 | `[total_tokens, heads, dim]` | `[num_seqs, 1, heads, dim]` |
