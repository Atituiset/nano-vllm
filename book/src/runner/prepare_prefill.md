# 4.3 Prefill 数据准备

`prepare_prefill` 是 prefill 阶段最复杂的数据准备函数——它需要将多个不定长序列打包成 FlashAttention 变长格式。

## 逐段解析 (`model_runner.py:129-170`)

### 1. 构建输入数据

```python
for seq in seqs:
    start = seq.num_cached_tokens           # 已缓存的 token 数（前缀缓存命中部分跳过）
    seqlen_q = seq.num_scheduled_tokens     # 本轮需要计算的 token 数
    end = start + seqlen_q
    seqlen_k = end                          # K/V 的长度 = 已缓存的 + 本轮计算的

    input_ids.extend(seq[start:end])        # 本轮的 token id
    positions.extend(range(start, end))     # 位置编码
```

**关键**：`seqlen_q` ≠ `seqlen_k`。前缀缓存命中时，`seqlen_k = end > seqlen_q`，FlashAttention 会从 KV Cache 中读取已缓存的 token。

### 2. 变长序列描述符

```python
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    max_seqlen_q = max(seqlen_q, max_seqlen_q)
    max_seqlen_k = max(seqlen_k, max_seqlen_k)
```

`cu_seqlens` 是累积序列长度——FlashAttention 用它来定位每个序列在扁平化输入中的起止位置。

例如：两个序列，seqlen_q 分别为 100 和 200：
```
cu_seqlens_q = [0, 100, 300]
```
表示序列0在 `input_ids[0:100]`，序列1在 `input_ids[100:300]`。

### 3. Slot Mapping 计算

```python
    start_block = start // self.block_size
    end_block = (end + self.block_size - 1) // self.block_size
    for i in range(start_block, end_block):
        slot_start = seq.block_table[i] * self.block_size
        if i == start_block:
            slot_start += start % self.block_size
        if i != end_block - 1:
            slot_end = seq.block_table[i] * self.block_size + self.block_size
        else:
            slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
        slot_mapping.extend(range(slot_start, slot_end))
```

**Slot Mapping** 是 KV Cache 写入的核心索引——它告诉 Triton kernel 每个 token 的 K/V 应该写入 KV Cache 张量的哪个 slot。

公式：`slot = block_id × block_size + offset_within_block`

例如 `block_table = [3, 7]`，`block_size = 4`，`start = 2, end = 6`：
```
block 0 (part): slot 3×4+2=14, slot 3×4+3=15
block 1 (full): slot 7×4+0=28, slot 7×4+1=29
```

### 4. Block Tables（前缀缓存时才需要）

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

只有当前缀缓存导致 K 长度 > Q 长度时，才需要传 `block_tables` 给 FlashAttention——因为此时 FlashAttention 需要从 KV Cache 中读取 Q 没有覆盖的那些 token 的 K/V。

### 5. 数据转移到 GPU

```python
input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
```

**`pin_memory=True` + `cuda(non_blocking=True)`**：使用页锁定内存实现异步 CPU→GPU 传输，减少数据搬运延迟。
