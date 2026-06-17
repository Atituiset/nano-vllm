# 4.2 KV Cache 分配

## 显存预算计算 (`model_runner.py:103-121`)

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
    self.kv_cache = torch.empty(2, num_hidden_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

### 公式推导

```
可用显存 = total × gpu_memory_utilization - 非模型占用 - 模型临时张量峰值 + 模型当前占用
         = total × 0.9 - (total - free) - peak + current
         = total × 0.9 - used - peak + current
```

关键洞察：
- `used - current` = 已分配但不在使用的临时张量（PyTorch 缓存池已回收，但 CUDA 分配器未归还 GPU）
- `peak` = warmup 时模型临时张量的峰值
- warmup 后 `current` = 模型权重 + 常驻张量

因此：`可用显存 = total × 0.9 - (模型权重 + 临时峰值)`

### 每块字节数

```
block_bytes = 2(K+V) × L(层数) × B(block_size) × H(KV头数) × D(头维) × S(数据类型字节数)
```

Qwen3-0.6B 示例：
- 28 层，8 KV 头，64 维，BF16(2 bytes)
- `block_bytes = 2 × 28 × 256 × 8 × 64 × 2 = 14,680,064 bytes ≈ 14 MB`

8GB 显卡（`gpu_memory_utilization=0.9`）：
- 可用 ≈ 8 × 0.9 - 模型占用 ≈ 6.5 GB
- 约 6.5GB / 14MB ≈ **464 个物理块**

### KV Cache 张量形状

`torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`

即 `[K/V, layer, block, token, head, dim]`。

后面在 Attention 层中，每个层拿到自己的 K 和 V 切片：

```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

每个层的 Attention 模块直接持有对 KV Cache 张量的引用——不需要每步拷贝或索引。
