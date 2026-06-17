# 1.2 请求生命周期

一个请求（Sequence）从进入到完成，经历以下完整生命周期：

```
用户输入 prompt
    │
    ▼
┌─────────────┐
│  WAITING     │  ← 入队等待，尚未分配 KV Cache
└──────┬──────┘
       │ schedule() 决定 prefill
       ▼
┌─────────────┐
│  Prefill     │  ← 处理 prompt token，填充 KV Cache
│  (RUNNING)   │
└──────┬──────┘
       │ postprocess() 检测 prompt 处理完毕
       ▼
┌─────────────┐
│  Decode      │  ← 逐 token 自回归生成
│  (RUNNING)   │     每步生成 1 token，追加 KV Cache
└──────┬──────┘
       │ EOS 或 max_tokens 达到
       ▼
┌─────────────┐
│  FINISHED    │  ← 释放所有 KV Cache 块
└─────────────┘
```

## 代码追踪

### 1. 入队：`add_request` (`llm_engine.py:43`)

```python
def add_request(self, prompt, sampling_params):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)       # → waiting 队列
```

- 字符串 prompt 先分词为 token id 列表
- 创建 `Sequence` 对象，初始状态 `WAITING`
- 挂入 Scheduler 的 `waiting` 双端队列

### 2. 调度循环：`generate` (`llm_engine.py:60`)

```python
while not self.is_finished():
    output, num_tokens = self.step()
```

`step()` 是单步迭代的核心：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()        # ① 调度
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 执行
    self.scheduler.postprocess(seqs, token_ids, is_prefill)       # ③ 后处理
```

### 3. Prefill 调度 (`scheduler.py:30-52`)

当 `waiting` 队列非空时，Scheduler 逐个检查：

- BlockManager 是否有足够空闲块？
- 当前批次 token 数是否超出 `max_num_batched_tokens`？
- 前缀缓存命中情况如何？

若全部 prompt token 都可以在本轮处理完，Sequence 从 `waiting` 移到 `running`。

### 4. Decode 调度 (`scheduler.py:58-73`)

当没有新的 prefill 请求时，对 `running` 队列中的序列逐个：

- 检查 BlockManager 是否有空闲块（新 token 可能需要新块）
- 若显存不足，执行**抢占**（preempt）：驱逐最后进入 running 的序列，释放其块
- 每个序列调度 1 个 token

### 5. 后处理 (`scheduler.py:81-92`)

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)   # 更新块的哈希（用于前缀缓存）
        seq.num_cached_tokens += seq.num_scheduled_tokens
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue       # chunked prefill: 还没处理完所有 prompt token
        seq.append_token(token_id)
        if token_id == eos or 达到 max_tokens:
            seq.status = FINISHED
            self.block_manager.deallocate(seq)  # 释放全部块
```

### 6. 关键状态转移

| 转移 | 触发条件 | 动作 |
|------|----------|------|
| WAITING → RUNNING | prefill 完成 prompt 全部 token | 移入 running 队列 |
| RUNNING → FINISHED | 生成 EOS 或达到 max_tokens | deallocate 释放块 |
| RUNNING → WAITING | 被抢占（显存不足） | deallocate 释放块，回到 waiting 队头 |
