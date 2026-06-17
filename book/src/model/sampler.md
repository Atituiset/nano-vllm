# 6.4 采样策略

## Sampler 实现 (`layers/sampler.py`)

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

## Gumbel-like 采样解析

最后一行看起来很魔幻，拆解如下：

### 数学等价

```
sample_tokens = argmax(Gumbel / temperature)
```

其中 `Gumbel = -log(Exponential(1))`。

### 逐步分解

1. `torch.empty_like(probs).exponential_(1)`：生成标准指数分布随机数 `U ~ Exp(1)`
2. `.clamp_min_(1e-10)`：避免零值（`-log(0) = inf`）
3. `probs.div_(U)`：`probs / U`
4. `.argmax(dim=-1)`：取最大值索引

### 推导

```
probs_i / U_i = exp(logit_i / T) / U_i
             = exp(logit_i / T - log(U_i))
             = exp(logit_i / T + Gumbel_i)     (Gumbel = -log(U))
```

对每个 token 取 `argmax` 等价于 `argmax(logit_i / T + Gumbel_i)`，即 **Gumbel-Max 采样**。

### 温度控制

- `temperature → 0`：logits 被 `div_` 放大，趋近 greedy（argmax logits）
- `temperature → ∞`：logits 被 `div_` 缩小到接近均匀分布，随机采样
- `temperature = 1.0`：标准随机采样

nano-vLLM 不支持 greedy decoding（`temperature > 1e-10` 的约束），因为 Gumbel-Max 在 `T=0` 时不稳定。

### 与标准 vLLM 采样的对比

| | vLLM | nano-vLLM |
|--|------|-----------|
| 方法 | 多项式采样（`torch.multinomial`） | Gumbel-Max 采样 |
| Greedy 支持 | ✅ `argmax` | ❌ |
| Top-p / Top-k 支持 | ✅ | ❌ |
| 计算效率 | `multinomial` 需要 CDF 计算 | `argmax` 单次扫描，更高效 |
| `@torch.compile` | - | ✅ 自动融合 |
