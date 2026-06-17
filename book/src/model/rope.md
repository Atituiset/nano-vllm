# 6.3 旋转位置编码

## 原理

RoPE（Rotary Position Embedding）通过将位置信息编码为旋转矩阵，使 Q·K 的点积自然包含相对位置信息。

### 数学表达

对于位置 `t` 的向量 `[x₁, x₂]`：

```
[x₁'] = [cos(tθ₁), -sin(tθ₁)] [x₁]
[x₂']   [sin(tθ₁),  cos(tθ₁)] [x₂]
```

实际实现将向量分成前后两半，而非相邻元素对。

## 实现 (`layers/rotary_embedding.py`)

### 预计算 cos/sin 表 (`rotary_embedding.py:29-35`)

```python
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
self.register_buffer("cos_sin_cache", torch.cat((freqs.cos(), freqs.sin(), dim=-1).unsqueeze_(1)))
```

`cos_sin_cache` 形状：`[max_position, 1, rotary_dim]`

### 应用 RoPE (`rotary_embedding.py:6-14`)

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)    # 后半部分
    y1 = x1 * cos - x2 * sin                       # 旋转实部
    y2 = x2 * cos + x1 * sin                       # 旋转虚部
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

### Forward（`@torch.compile`）(`rotary_embedding.py:37-48`)

```python
@torch.compile
def forward(self, positions, query, key):
    cos_sin = self.cos_sin_cache[positions]    # 按位置索引
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return query, key
```

`@torch.compile` 让 PyTorch 编译器优化这个函数，自动融合多个 kernel。

### 索引方式

`positions` 是一个 1D 张量（prefill 时长度 = 总 token 数，decode 时长度 = batch size），直接用于索引预计算的 cos/sin 表。不同 token 可以有不同位置——支持 chunked prefill 和前缀缓存场景。
