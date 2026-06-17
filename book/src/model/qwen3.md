# 6.1 Qwen3 模型结构

## 层级结构 (`models/qwen3.py`)

```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding      # 词嵌入
│   ├── Qwen3DecoderLayer × N       # 解码器层
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── Qwen3Attention
│   │   │   ├── QKVParallelLinear   # Q/K/V 投影
│   │   │   ├── q_norm, k_norm (RMSNorm)  # Qwen3 特有：QK Norm
│   │   │   ├── RotaryEmbedding      # RoPE
│   │   │   ├── Attention            # FlashAttention + KV Cache
│   │   │   └── RowParallelLinear    # O 投影
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   └── Qwen3MLP
│   │       ├── MergedColumnParallelLinear  # Gate + Up 投影
│   │       ├── SiluAndMul                 # 门控激活
│   │       └── RowParallelLinear          # Down 投影
│   └── RMSNorm                    # 最终 LayerNorm
└── ParallelLMHead                 # 输出投影
```

## Qwen3DecoderLayer 的残差连接 (`qwen3.py:146-159`)

```python
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

**融合残差设计**：RMSNorm 同时做归一化和残差加法（`add_rms_forward`），而不是先加残差再做 Norm。这节省了一次额外的显存读写。

```
标准实现:   residual = x + attn_output;  y = norm(residual)
融合实现:   y, residual = norm(x + residual_old, residual_old)  # 一次操作同时输出归一化值和新残差
```

## Qwen3 的 QK Norm

Qwen3 在 `qkv_bias=False` 时额外添加 Q 和 K 的 RMSNorm：

```python
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

这是 Qwen3 架构的特色——在 QKV 投影后对 Q 和 K 分别做 RMSNorm，有助于稳定大模型的训练。
