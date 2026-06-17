# 7.3 Config 与参数推导

## Config 定义 (`config.py`)

```python
@dataclass(slots=True)
class Config:
    model: str                                    # 模型路径
    max_num_batched_tokens: int = 16384           # 单步最大 token 数
    max_num_seqs: int = 512                       # 最大并行序列数
    max_model_len: int = 4096                     # 最大模型序列长度
    gpu_memory_utilization: float = 0.9           # GPU 显存利用率
    tensor_parallel_size: int = 1                 # 张量并行数
    enforce_eager: bool = False                   # 禁用 CUDA Graph
    hf_config: AutoConfig | None = None           # HuggingFace 模型配置
    eos: int = -1                                 # EOS token ID
    kvcache_block_size: int = 256                 # KV Cache 块大小
    num_kvcache_blocks: int = -1                  # 物理块数量（运行时推导）
```

## 参数推导链

```
model路径 → AutoConfig → hf_config
                │
                ├→ max_position_embeddings──→ min(max_model_len, max_position_embeddings)
                │
                ├→ num_hidden_layers ──┐
                ├→ num_key_value_heads──┤→ block_bytes → num_kvcache_blocks
                ├→ head_dim ────────────┤    (在 allocate_kv_cache 中)
                ├→ hidden_size ─────────┤
                └→ dtype.itemsize ───────┘
```

### `__post_init__` (`config.py:20-25`)

```python
def __post_init__(self):
    assert os.path.isdir(self.model)
    assert self.kvcache_block_size % 256 == 0
    assert 1 <= self.tensor_parallel_size <= 8
    self.hf_config = AutoConfig.from_pretrained(self.model)
    self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

- 验证模型路径存在
- 验证 block_size 是 256 的倍数（对齐 FlashAttention 的 block）
- 限制 TP ≤ 8（NCCL 通信效率上限）
- `max_model_len` 不超过模型支持的最大位置编码长度

### `num_kvcache_blocks` 的延迟推导

`num_kvcache_blocks` 初始值为 -1，在 `ModelRunner.allocate_kv_cache()` 中根据实际显存情况动态推导——这是一个**运行时才能确定的参数**，因为模型权重和临时张量的显存占用只有在 warmup 之后才准确。

## 从 LLM 到 Config 的参数筛选 (`llm_engine.py:17-20`)

```python
def __init__(self, model, **kwargs):
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(model, **config_kwargs)
```

LLM 的 `__init__` 只将合法的 Config 字段传递下去——用户传入的无关参数（如 vLLM 特有的参数）会被静默忽略。
