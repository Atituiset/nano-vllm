# 3.3 Decode 调度策略

当没有新请求需要 prefill 时，Scheduler 进入 decode 模式，为 `running` 队列中的序列逐个分配执行资源。

## 逐段代码解析 (`scheduler.py:57-73`)

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
```

## 逐步解读

### 1. 遍历 running 队列

从 `running` 队列头部逐个取出序列，尝试为其分配 decode 所需的资源。

### 2. 检查是否需要新块

`can_append(seq)` 返回 `len(free_block_ids) >= (len(seq) % block_size == 1)`。

只有当新 token 需要一个新块时（序列长度刚好是 block_size 的整数倍 + 1）才需要检查空闲块是否足够。

### 3. 显存不足时的抢占

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())  # 抢占队尾序列
    else:
        self.preempt(seq)                  # 抢占自己
        break
```

**抢占策略**：从 `running` 队列**尾部**开始抢占（LIFO——最后进来的最先被踢出），直到腾出足够的物理块。

如果 `running` 队列空了还不够，就把当前序列自己也抢占。

### 4. Python `while-else` 语法

这里的 `else` 块与 `while` 配对：**只有 while 循环正常结束（未被 break）时才执行 else**。

即：如果 `can_append(seq)` 一开始就返回 True，或者通过抢占腾出了足够空间后返回 True，进入 else 分支设置 `num_scheduled_tokens=1` 并分配新块。

如果抢占了自己（`break`），则跳过 else，该序列被放回 `waiting`。

### 5. 恢复 running 队列

```python
self.running.extendleft(reversed(scheduled_seqs))
```

调度完的序列按原顺序插回 `running` 队列头部。`extendleft + reversed` 保证序列的原始相对顺序不变。

### 6. 断言保证

```python
assert scheduled_seqs
```

decode 阶段**必须**有序列被调度——如果 running 队列非空却无法调度任何序列，说明存在逻辑错误。
