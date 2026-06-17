# 8.1 权重分片加载

## 核心函数 (`utils/loader.py`)

```python
def load_model(model, path):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

## 加载流程

```
1. 遍历模型目录下的所有 .safetensors 文件
2. 对每个权重名：
   a. 检查是否匹配 packed_modules_mapping 中的基名
      - 是：替换基名→合并名，传 shard_id 给 weight_loader
   b. 否：直接用 default_weight_loader 拷贝
```

## weight_loader 协议

每个 `nn.Parameter` 可以挂载一个 `weight_loader` 方法，控制该参数如何从 HuggingFace 权重文件加载数据。

| 层类型 | weight_loader | 行为 |
|--------|---------------|------|
| `ReplicatedLinear` | 直接 `copy_` | 完整拷贝，不做分片 |
| `ColumnParallelLinear` | 按 `tp_rank` 切片输出维 | 只拷属于本 rank 的列 |
| `RowParallelLinear` | 按 `tp_rank` 切片输入维 | 只拷属于本 rank 的行 |
| `QKVParallelLinear` | 按 shard_id("q"/"k"/"v") 切片 | Q/K/V 各自按 tp 切分 |
| `MergedColumnParallelLinear` | 按 shard_id(0/1) 切片 | Gate/Up 各自按 tp 切分 |
| `VocabParallelEmbedding` | 按 `tp_rank` 切片词表维 | 只拷属于本 rank 的词表区间 |

## 注意

权重加载在 **CPU** 上执行（`safe_open(file, "pt", "cpu")`），然后通过 `copy_` 拷贝到 GPU 上的参数——避免在 GPU 上分配临时张量浪费显存。
