# 6.2 张量并行线性层

tensor 并行（TP）的核心思想：将一个大矩阵切分到多张 GPU 上，每张卡只存 1/N 的权重、算 1/N 的计算量。

## 层级体系 (`layers/linear.py`)

```
LinearBase (基类)
├── ReplicatedLinear         # 不切分，每卡完整副本
├── ColumnParallelLinear     # 输出维切分
│   ├── QKVParallelLinear   # Q/K/V 分别切分
│   └── MergedColumnParallelLinear  # 多组输出分别切分后合并
└── RowParallelLinear         # 输入维切分
```

## ColumnParallelLinear (`linear.py:54-73`)

**切分方式**：输出维按 `tp_size` 等分，每卡持有 `output_size / tp_size` 列。

```
完整权重: [hidden, output]     → 每卡: [hidden, output/tp]
输入 x:   [*, hidden]          → 输出:   [*, output/tp]
无需通信（列切分后各卡独立计算）
```

GQA 注意力示例（`tp=2`, `num_heads=4`, `num_kv_heads=2`）：
```
Rank 0: Q头0-1, K头0, V头0
Rank 1: Q头2-3, K头1, V头1
```

## QKVParallelLinear (`linear.py:96-128`)

QKV 投影的特殊性：Q 和 K/V 的头数不同（GQA），需要分别计算各自的分片大小和偏移。

```python
def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False):
    tp_size = dist.get_world_size()
    self.num_heads = total_num_heads // tp_size
    self.num_kv_heads = total_num_kv_heads // tp_size
    output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
```

权重布局：`[Q分片 | K分片 | V分片]`

`weight_loader` 根据 `loaded_shard_id`（"q"/"k"/"v"）计算各分片的偏移和大小：

```python
if loaded_shard_id == "q":
    shard_size = self.num_heads * self.head_size     # Q 的头数 × 头维
    shard_offset = 0
elif loaded_shard_id == "k":
    shard_size = self.num_kv_heads * self.head_dim   # KV 头数 × 头维
    shard_offset = self.num_heads * self.head_size
else:  # "v"
    shard_size = self.num_kv_heads * self.head_dim
    shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
```

## MergedColumnParallelLinear (`linear.py:76-93`)

Gate + Up 合并投影：将两个独立的线性层 `(hidden→intermediate)` 各自切分后合并为一个矩阵乘。

```python
# gate_up_proj: [hidden, 2 × intermediate/tp]
# 前 half = gate, 后 half = up
```

## RowParallelLinear (`linear.py:131-156`)

**切分方式**：输入维按 `tp_size` 等分，每卡持有 `input_size / tp_size` 行。

```
完整权重: [input, output] → 每卡: [input/tp, output]
输入 x:   [*, input/tp]   → 输出: [*, output]
需要 all_reduce ！（每卡算出部分结果后求和）
```

```python
def forward(self, x):
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        dist.all_reduce(y)
    return y
```

注意：`bias` 只在 rank 0 上加——all_reduce 后每卡的结果相同，只在 rank 0 加一次避免重复。

## TP 通信模式

```
[ColumnParallel] → [RowParallel] 之间无需通信
      ↑                    |
  独立输出          all_reduce 求和
      |                    ↓
   QKV 投影              O 投影 / Down 投影
```

经典的 "Column-then-Row" 模式：列并行 + 行并行之间天然形成一次 all_reduce 通信。
