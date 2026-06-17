# 8.2 权重打包映射

## `packed_modules_mapping` (`qwen3.py:187-193`)

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":  ("gate_up_proj", 1),
    }
```

## 为什么需要打包？

HuggingFace 格式的权重是分离存储的：

```
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
```

但推理时，将 QKV 合并为一次矩阵乘法更高效（`QKVParallelLinear`），gate 和 up 同理（`MergedColumnParallelLinear`）。

## 映射工作流

当 loader 遇到 `model.layers.0.self_attn.q_proj.weight` 时：

1. 匹配 `packed_modules_mapping` 中的键 `"q_proj"`
2. 得到值 `("qkv_proj", "q")`
3. 将权重名中的 `q_proj` 替换为 `qkv_proj`
4. 在模型中查找 `model.layers.0.self_attn.qkv_proj.weight`
5. 调用 `weight_loader(param, loaded_weight, shard_id="q")`
6. `QKVParallelLinear.weight_loader` 根据 `shard_id="q"` 计算偏移，将 Q 权重写入合并矩阵的正确位置

```
加载前 qkv_proj.weight:
[  随机初始化  ]

加载 q_proj → 写入偏移0:               [Q_shard |  空  |  空  ]
加载 k_proj → 写入偏移Q_size:          [Q_shard |K_shard|  空  ]
加载 v_proj → 写入偏移Q_size+K_size:   [Q_shard |K_shard|V_shard]
```

## 优势

- **减少 kernel launch**：一次 GEMM 替代三次
- **更好的内存局部性**：QKV 权重连续存储
- **TP 通信优化**：Column-then-Row 模式天然嵌入
