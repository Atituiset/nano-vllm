# 3.2 Prefill 调度策略

## 逐段代码解析 (`scheduler.py:30-52`)

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]                           # 看队头
    remaining = self.max_num_batched_tokens - num_batched_tokens  # 剩余 token 额度
    if remaining == 0:
        break

    if not seq.block_table:                         # 尚未分配块（新序列）
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:                 # 显存不足
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # 实际需计算的 token
    else:
        num_tokens = seq.num_tokens - seq.num_cached_tokens  # chunked prefill 续传

    if remaining < num_tokens and scheduled_seqs:   # 放不下且非首个
        break

    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)

    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens

    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)

    scheduled_seqs.append(seq)
```

## 关键决策点

### 1. 前缀缓存跳过计算

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

如果 1024 token 的 prompt 中前 512 token（2 个块）命中缓存，则只需计算 `1024 - 512 = 512` 个 token。这直接减少了 prefill 的计算量。

### 2. 序列是否移入 running

```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
```

只有**本轮就能处理完所有 prompt token** 的序列才移入 running。否则序列仍留在 `waiting` 中，下轮继续 prefill（Chunked Prefill）。

### 3. 只对第一个序列允许 Chunked Prefill

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

如果当前 token 额度放不下这个序列的全部 token：
- 如果是**第一个序列**（`scheduled_seqs` 为空），允许拆分
- 如果**已有其他序列**，跳过——不让后续序列挡住 chunked 序列

### 4. 新序列的 block 分配时机

```python
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
```

分配块在**计算 num_tokens 之后、设置 num_scheduled_tokens 之前**。这确保了：
- 先确定有多少缓存可复用（`can_allocate`）
- 再正式分配物理块（`allocate`）
- 最后决定本轮调度多少 token
