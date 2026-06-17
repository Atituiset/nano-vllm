# 4.6 Tensor Parallelism 通信

nano-vLLM 的张量并行（TP）通过 **spawn 多进程 + NCCL + 共享内存 RPC** 实现。

## 架构

```
┌─────────────────────────────────────┐
│           LLMEngine (rank 0)       │
│  ┌─────────┐  ┌─────────────────┐  │
│  │Scheduler│  │ModelRunner(rank0)│  │
│  └─────────┘  └───────┬─────────┘  │
│                       │ SharedMem  │
└───────────────────────┼────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼                               ▼
┌───────────────┐              ┌───────────────┐
│ModelRunner    │              │ModelRunner    │
│(rank 1)       │              │(rank N-1)     │
│Event.wait()──│──Event.set()─│               │
│loop → read   │              │loop → read   │
│shm → call    │              │shm → call    │
└───────────────┘              └───────────────┘
```

## 主进程端 (`llm_engine.py:24-30`, `model_runner.py:76-83`)

```python
# LLMEngine.__init__
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()

# ModelRunner.write_shm
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()
```

## Worker 进程端 (`model_runner.py:61-73`)

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break

def read_shm(self):
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
```

### 通信流程

1. Rank 0 将方法名和参数序列化到共享内存
2. Rank 0 通过 `Event.set()` 唤醒所有 worker
3. Worker 读取共享内存，反序列化参数
4. Worker 清除 Event（`event.clear()`），执行方法
5. NCCL `all_reduce` / `gather` 在方法内部完成同步

### 为什么用共享内存而不是 RPC？

- 共享内存是**零拷贝**的——数据直接在进程间共享
- 对于大参数（如序列列表），避免了多次序列化/网络传输
- NCCL 本身就需要进程间通信，共享内存只是辅助控制流

## `call` 的统一入口 (`model_runner.py:85-89`)

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)    # 通知 workers
    method = getattr(self, method_name, None)
    return method(*args)                       # 自己也执行
```

Rank 0 既负责通知 worker，自己也执行方法。这保证所有 rank 同步执行同一个方法。
