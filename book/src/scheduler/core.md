# 3.1 Scheduler 核心逻辑

## 数据结构 (`scheduler.py:10-17`)

```python
class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs           # 最大并行序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单步最大 token 数
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(...)
        self.waiting: deque[Sequence] = deque()   # 等待队列
        self.running: deque[Sequence] = deque()   # 运行队列
```

两个双端队列构成了最简调度模型：

- `waiting`：尚未处理或被抢占的序列
- `running`：正在 decode 的序列

### 与 vLLM 的差异

vLLM 有三个队列：`waiting`、`running`、`swapped`（换出到 CPU 内存的序列）。nano-vLLM 没有 swapping，被抢占的序列直接回 `waiting` 队列头，需要**重新计算**被释放的 KV Cache。

## `schedule()` 方法的整体结构 (`scheduler.py:25-73`)

```python
def schedule(self):
    scheduled_seqs = []
    num_batched_tokens = 0

    # === 优先做 Prefill ===
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        # 尝试将 waiting 中的序列调度进来 ...
    if scheduled_seqs:
        return scheduled_seqs, True    # is_prefill=True

    # === 没有 Prefill，做 Decode ===
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        # 尝试将 running 中的序列调度进来 ...
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False       # is_prefill=False
```

**关键设计**：Prefill 和 Decode **互斥**——同一步要么做所有 prefill，要么做所有 decode。不做混合批处理（chunked prefill 是特例，详见 3.5）。

这个设计简化了模型执行路径：
- Prefill 走 `flash_attn_varlen_func`（变长序列注意力）
- Decode 走 `flash_attn_with_kvcache`（带 KV Cache 的注意力）

## 两个调度约束

1. **`max_num_seqs`**：单步最多处理 512 个序列（受 FlashAttention 的 block 大小限制）
2. **`max_num_batched_tokens`**：单步最多处理 16384 个 token（受 GPU 显存和计算量限制）

Prefill 受 token 数约束更严（每个序列可能几千 token），Decode 受序列数约束更严（每序列只 1 token）。
