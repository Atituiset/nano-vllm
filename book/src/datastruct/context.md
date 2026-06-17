# 7.2 Context 全局上下文

`Context` 是一个全局单例，存储当前 forward pass 的一切元数据——所有层共享同一个 Context 实例。

## 定义 (`utils/context.py`)

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None      # Prefill 专用：Q 的累积序列长度
    cu_seqlens_k: torch.Tensor | None = None      # Prefill 专用：K 的累积序列长度
    max_seqlen_q: int = 0                          # Prefill 专用：Q 的最大序列长度
    max_seqlen_k: int = 0                          # Prefill 专用：K 的最大序列长度
    slot_mapping: torch.Tensor | None = None       # Prefill + Decode：KV Cache 写入索引
    context_lens: torch.Tensor | None = None       # Decode 专用：每个序列的上下文长度
    block_tables: torch.Tensor | None = None       # Prefill(前缀缓存) + Decode：物理块映射表

_CONTEXT = Context()
```

## 全局变量模式

```python
def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, ...):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, ...)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

这是一个**线程不安全**的设计——依赖 Python GIL 保证同一次 forward 内只有一个调用在执行。

## 使用场景

| 阶段 | 设置者 | 消费者 |
|------|--------|--------|
| `is_prefill` | `ModelRunner.prepare_*()` | `Attention.forward()`, `ParallelLMHead.forward()` |
| `cu_seqlens_q/k` | `prepare_prefill()` | `flash_attn_varlen_func()` |
| `slot_mapping` | `prepare_prefill()` / `prepare_decode()` | `store_kvcache_kernel()` |
| `context_lens` | `prepare_decode()` | `flash_attn_with_kvcache()` |
| `block_tables` | `prepare_prefill()`(前缀缓存时) / `prepare_decode()` | `flash_attn_*()` |

## 为什么用全局变量而不是传参？

1. **简化接口**：`Attention.forward(q, k, v)` 不需要额外传一大堆元数据参数
2. **模型层透明**：`Qwen3Model.forward(input_ids, positions)` 内部遍历各层，不需要把元数据逐层传递
3. **CUDA Graph 兼容**：图内代码无法接受 Python 参数，但可以读全局张量

**vLLM 的做法**：vLLM 将类似信息封装在 `AttentionMetadata` 中，通过参数传递。两者功能等价，nano 的做法更简洁但扩展性较差。
