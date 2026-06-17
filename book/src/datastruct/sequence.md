# 7.1 Sequence 状态机

`Sequence` 是 nano-vLLM 中最基础的数据结构——它表示一个推理请求的完整生命周期。

## 定义 (`engine/sequence.py`)

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 等待 prefill
    RUNNING = auto()    # 正在 decode
    FINISHED = auto()   # 生成完毕

class Sequence:
    block_size = 256                    # 类变量，由 Config 设置
    counter = count()                   # 全局自增 ID 生成器

    def __init__(self, token_ids, sampling_params=SamplingParams()):
        # 身份
        self.seq_id = next(Sequence.counter)

        # 状态
        self.status = SequenceStatus.WAITING
        self.is_prefill = True

        # Token 缓冲
        self.token_ids = copy(token_ids)    # 深拷贝，避免外部修改
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)

        # 调度计数器
        self.num_cached_tokens = 0          # KV Cache 中已有的 token 数
        self.num_scheduled_tokens = 0       # 本轮被调度执行的 token 数

        # Block Table
        self.block_table = []               # 逻辑块索引 → 物理块 ID

        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

## 状态转移图

```
         add_request()
WAITING ◄────────────────────────┐
  │                                │
  │ schedule(prefill完成)          │ preempt()
  ▼                                │
RUNNING ──────────────────────────┘
  │
  │ postprocess(EOS/max_tokens)
  ▼
FINISHED → deallocate()
```

## 关键属性推导

### `num_blocks`（`sequence.py:56-57`）

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

向上取整。一个序列可能不满一个完整块，但仍需要一整个物理块。

### `last_block_num_tokens`（`sequence.py:60-61`）

```python
@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

最后一个块中的有效 token 数。用于 decode 阶段计算 slot mapping。

### `block(i)`（`sequence.py:63-65`）

```python
def block(self, i):
    return self.token_ids[i * self.block_size: (i+1) * self.block_size]
```

第 i 个逻辑块对应的 token id 切片。前缀缓存哈希计算中使用。

## 序列化（多进程传递）

`__getstate__` / `__setstate__` 控制 pickle 序列化行为——用于 Tensor Parallelism 的共享内存 RPC。

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)
```

**关键优化**：

- Prefill 时传递完整 `token_ids`（因为模型需要处理所有 prompt token）
- Decode 时只传 `last_token`（一个 int，而非整个序列）——大幅减少共享内存写入量

这个序列化设计是非常精妙的优化：decode 时序列可能有几千个 token，但模型只需要最后一个 token 的 ID。
