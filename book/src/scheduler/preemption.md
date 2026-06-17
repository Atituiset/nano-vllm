# 3.4 抢占与驱逐

## 抢占机制 (`scheduler.py:75-79`)

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

抢占一个序列做了三件事：

1. **状态回退**：`RUNNING → WAITING`
2. **标记为需要重新 prefill**：`is_prefill = True`
3. **释放全部物理块**：`deallocate` 释放该序列的所有 KV Cache 块
4. **回到等待队列头部**：`appendleft` 保证被抢占的序列下次优先获得调度

## 代价：重计算（Recomputation）

nano-vLLM 没有实现 Swapping（将 KV Cache 拷到 CPU 内存再拷回）。被抢占的序列：

- 释放所有 KV Cache → GPU 显存腾出
- 回到 `waiting` → 下次调度时重新执行 prefill
- 前缀缓存可能部分命中 → 减少量计算，但被释放的非共享块需要全部重算

**与 vLLM 的区别**：

| | nano-vLLM | vLLM |
|--|-----------|------|
| 抢占策略 | 释放块 + 重计算 | 优先换出到 CPU（swap），空间还不够才释放块（recompute） |
| 代价 | 重计算 prompt | 换出/换入延迟 < 重计算延迟，但占 CPU 内存 |
| 实现复杂度 | 简单 | 需要 CPU pinned memory、swap in/out 调度 |

## LIFO 抢占顺序

```python
self.preempt(self.running.pop())  # 从队尾抢占
```

为什么选择 LIFO？

- **公平性**：刚进入 running 的序列贡献最少，优先牺牲
- **Prefix Cache 友好**：早进入 running 的序列可能有更多共享前缀，保留它们能最大化缓存命中率
- **简化逻辑**：不需要估算每个序列的"价值"来做最优决策

## 极端情况的守卫

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)    # 抢占自己
        break
```

如果队列空了还不够，抢占当前序列自己。此序列回到 `waiting` 队头，下一步调度时：
- `can_allocate` 检查空闲块
- 如果腾出的块足够，直接重新 prefill
- 如果还不够，每轮逐步通过抢占其他序列积累空闲块
