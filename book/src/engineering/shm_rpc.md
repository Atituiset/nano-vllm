# 8.3 共享内存 RPC

## 通信架构

```
┌─────────────────────────────────┐
│ Rank 0 (主进程)                  │
│                                  │
│  self.call("run", seqs, True)   │
│    ├→ write_shm("run", seqs, True)   # 1. 序列化参数到共享内存
│    ├→ event[0].set()                 # 2. 唤醒 worker 1
│    ├→ event[1].set()                 # 3. 唤醒 worker 2
│    └→ self.run(seqs, True)           # 4. 自己也执行
│                                  │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Rank N (Worker 进程)             │
│                                  │
│  while True:                     │
│    event.wait()                  # 等待 Rank 0 唤醒
│    read_shm()                    # 读取共享内存
│    self.run(seqs, True)          # 执行
│    event.clear()                 # 清除信号
│                                  │
└─────────────────────────────────┘
```

## 共享内存协议 (`model_runner.py:68-83`)

```python
def read_shm(self):
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")     # 前 4 字节：数据长度
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 剩余：pickle 数据
    self.event.clear()
    return method_name, args

def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()
```

### 协议格式

```
┌──────────┬─────────────────────────┐
│ 4 bytes  │     n bytes             │
│ len = n  │  pickle([method, args]) │
└──────────┴─────────────────────────┘
```

### 共享内存大小

```python
self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1 MB
```

1MB 对于序列元数据足够了——因为 decode 时 `Sequence.__getstate__` 只传 `last_token`（一个 int）。

## Event 同步

```python
event = ctx.Event()    # multiprocessing.Event → 系统级信号量
```

- `event.set()`：唤醒 worker
- `event.wait()`：阻塞等待唤醒
- `event.clear()`：重置为未唤醒状态

**时序保证**：

1. Rank 0 先写共享内存，再 set event → 保证 worker 读取时数据已就绪
2. Worker 先读共享内存，再 clear event → 保证数据读完后才允许下次写入

## 局限性

1. 共享内存大小固定（1MB）——超长 prefill token 列表可能溢出
2. 无流量控制——Rank 0 比 worker 快时可能覆盖未读取的数据
3. 仅支持 `spawn` 上下文——`fork` 可能导致 CUDA 状态不一致
