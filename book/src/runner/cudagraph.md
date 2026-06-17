# 4.5 CUDA Graph 捕获

CUDA Graph 是 nano-vLLM 在 decode 阶段消除 kernel launch overhead 的核心技术。

## 问题：Decode 的 kernel launch overhead

Decode 每步只处理 1 个 token per sequence，模型 forward 涉及数十个 CUDA kernel（矩阵乘、LayerNorm、Attention 等）。每个 kernel 的 launch 有 ~5-10μs 的 CPU 开销。当 GPU 计算量很小（只 512 个序列各 1 token）时，launch overhead 可能占总时间的 30-50%。

## 解决方案：CUDA Graph

CUDA Graph 将整个模型 forward 的所有 kernel 录制为一幅"执行图"，后续只需 `graph.replay()` 即可一次性提交所有 kernel，消除逐个 launch 的 CPU 开销。

## 捕获过程 (`model_runner.py:222-257`)

```python
@torch.inference_mode()
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # 预分配捕获用的固定形状张量
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # 分桶：1, 2, 4, 8, 16, 32, 48, ..., 512
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], ...)
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
```

### 分桶策略

`graph_bs = [1, 2, 4, 8, 16, 32, 48, 64, ..., 512]`

CUDA Graph 要求输入形状与捕获时完全一致。为每个可能的 batch size 都捕获一个图太浪费，因此分桶：实际 batch size ≤ 桶大小即使用该桶的图。

查找代码 (`model_runner.py:202`)：

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

### 共享 graph_pool

```python
if self.graph_pool is None:
    self.graph_pool = graph.pool()
```

所有图共享同一个 `graph_pool`（CUDA 显存池），显著减少显存占用。第一个捕获的图的 pool 作为后续所有图的 pool。

### 逆序捕获

```python
for bs in reversed(self.graph_bs):
```

从大到小捕获——因为大 batch size 需要更多临时显存，先捕获大图可以确保 pool 分配足够的空间。小图复用大图的 pool 不会有问题。

## 重放过程 (`model_runner.py:197-212`)

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    bs = input_ids.size(0)
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    graph_vars = self.graph_vars
    graph_vars["input_ids"][:bs] = input_ids
    graph_vars["positions"][:bs] = positions
    graph_vars["slot_mapping"].fill_(-1)
    graph_vars["slot_mapping"][:bs] = context.slot_mapping
    graph_vars["context_lens"].zero_()
    graph_vars["context_lens"][:bs] = context.context_lens
    graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 什么情况不用 CUDA Graph？

1. **Prefill**：序列长度每次不同，无法用固定形状的图
2. **`enforce_eager=True`**：用户显式禁用
3. **batch_size > 512**：超过了预捕获的最大桶

### 数据填充

CUDA Graph 工作在固定形状的张量上。每步只需将实际数据拷入预分配张量的前 `bs` 个位置，然后 `replay()`。`-1` 的 slot_mapping 会被 Triton kernel 跳过（`if slot == -1: return`）。
