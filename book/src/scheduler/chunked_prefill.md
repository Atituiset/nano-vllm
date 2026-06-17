# 3.5 Chunked Prefill

## 动机

当 prompt 非常长（如 4096 token）但 `max_num_batched_tokens` 只有 16384 时：

- 如果不拆分，一个长 prompt 就可能占满整个批次
- 拆分后，短 prompt 可以和长 prompt 的第一段一起 prefill
- 后续段在后续步骤中继续，直到整个 prompt 处理完毕

## nano-vLLM 的实现

Chunked Prefill 在 `scheduler.py:42-43` 中实现——只有两行核心逻辑：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

**含义**：

1. 只允许**第一个序列**做 chunked prefill
2. 后续序列要么整个放进来，要么等下一轮

### 第一个序列的 chunked prefill

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

当一个序列需要的 token 数 `num_tokens` 超过剩余额度 `remaining` 时，只调度 `remaining` 个 token。

此时序列**不满足** `num_cached_tokens + num_scheduled_tokens == num_tokens` 条件，所以它**留在 `waiting` 队列**，不移入 `running`。

### 下一轮调度

下一轮 `schedule()` 被调用时：

```python
if not seq.block_table:    # 第一次进来的新序列
    ...
else:                       # 已分配块，是 chunked prefill 续传
    num_tokens = seq.num_tokens - seq.num_cached_tokens
```

已分配块的序列直接计算剩余需要处理的 token 数，不再调用 `can_allocate`（因为块已经分配好了）。

### 为什么只允许第一个序列？

如果第二个序列也允许拆分，可能导致：

- 多个未完成的 chunked 序列同时占用 `waiting` 队列前端
- 单步 prefill 的 token 预算被多个序列瓜分
- 每个序列都推进一点点，但都不能生成→吞吐量下降

只允许第一个序列 chunked，保证它尽快完成 prefill 进入 decode。
