# 2.1 物理块与逻辑块

## 核心思想

传统 LLM 推理为每个序列预分配一段**连续**的 KV Cache 显存。问题：

- 不同序列长度差异巨大 → **显存碎片化**
- 预分配最大长度 → **显存浪费**
- 无法共享相同前缀 → **重复计算与缓存**

vLLM 的解法：像操作系统管理虚拟内存一样管理 KV Cache：

| 操作系统 | LLM 推理框架 |
|----------|-------------|
| 虚拟地址空间 | 序列的逻辑 token 序列 |
| 页（Page） | 块（Block），大小 `block_size=256` tokens |
| 页表（Page Table） | block_table：逻辑块 → 物理块 ID |
| 物理帧（Frame） | GPU 显存上的一块 KV Cache 存储区域 |
| 缺页中断 | 按需分配新物理块 |
| 共享内存 | 前缀缓存（Prefix Caching） |

## 块大小的选择

`config.py:17` 中默认 `kvcache_block_size = 256`，且要求是 256 的倍数：

```python
assert self.kvcache_block_size % 256 == 0
```

选择 256 的原因：

1. FlashAttention 的 block 大小通常是 64 或 128，256 可以整除
2. 太小的块（如 16）导致 block_table 过长，增加 GPU 端查找开销
3. 太大的块（如 4096）导致内部碎片增多

## 逻辑块计算

`sequence.py:56-57`：

```python
@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size
```

向上取整：一个序列有 `n` 个 token，需要 `⌈n / block_size⌉` 个逻辑块。

## 逻辑块到物理块的映射

`Sequence` 中的 `block_table` 是一个列表，索引是逻辑块号，值是物理块 ID：

```python
seq.block_table = [3, 7, 12]
# 逻辑块0 → 物理块3
# 逻辑块1 → 物理块7
# 逻辑块2 → 物理块12
```

这种间接映射的好处：

1. **物理块不必连续**——消除了显存碎片化
2. **多个序列可共享同一物理块**——前缀缓存
3. **按需分配**——序列增长时只需追加新物理块

## 物理块在 GPU 上的存储

`model_runner.py:115`：

```python
self.kv_cache = torch.empty(
    2,                          # K 和 V
    num_hidden_layers,          # 每层各一份
    num_kvcache_blocks,         # 物理块总数
    block_size,                 # 每块 256 token
    num_kv_heads,               # KV 头数
    head_dim                    # 头维度
)
```

每个物理块的显存占用（`model_runner.py:112`）：

```
block_bytes = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_size
```

例如 Qwen3-0.6B（28层，8个KV头，64维头，BF16）：
```
block_bytes = 2 × 28 × 256 × 8 × 64 × 2 = 14,680,064 bytes ≈ 14 MB
```

1000 个物理块 ≈ 14 GB 显存用于 KV Cache。
