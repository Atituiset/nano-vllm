# 4.1 ModelRunner 初始化

## 完整初始化流程 (`model_runner.py:17-49`)

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        # 1. 初始化分布式
        dist.init_process_group("nccl", "tcp://localhost:2333", ...)
        torch.cuda.set_device(rank)

        # 2. 构建模型
        torch.set_default_dtype(hf_config.dtype)    # BF16
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)         # 加载权重

        # 3. 采样器
        self.sampler = Sampler()

        # 4. Warmup（触发 CUDA malloc 预分配）
        self.warmup_model()

        # 5. 根据剩余显存分配 KV Cache
        self.allocate_kv_cache()

        # 6. 捕获 CUDA Graph（decode 加速）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 7. 多卡时设置共享内存 RPC
        if self.world_size > 1 and rank > 0:
            self.loop()    # worker 卡进入循环等待
```

## 初始化顺序为什么重要

**warmup → allocate_kv_cache → capture_cudagraph** 这个顺序不可调换：

1. **warmup**：执行一次 dummy forward，触发 PyTorch 的 CUDA 内存缓存分配器。这一步的峰值显存使用量是模型自身 + 临时张量的上限
2. **allocate_kv_cache**：读取 warmup 后的显存状态，用 `total × gpu_memory_utilization - 模型占用` 来计算可分配的 KV Cache 块数
3. **capture_cudagraph**：CUDA Graph 需要 KV Cache 已经分配好，因为它要录制 decode 的完整执行流

如果先分配 KV Cache 再 warmup，模型临时张量可能超出预留空间。如果先捕获 CUDA Graph 再分配 KV Cache，录制时 KV Cache 未就绪。

## NCCL 初始化

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
```

即使单卡运行（`tensor_parallel_size=1`），也初始化了进程组——这使得所有张量并行的线性层代码不需要分 `if tp_size > 1` 分支。
