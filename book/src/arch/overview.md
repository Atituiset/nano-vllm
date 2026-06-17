# 1.1 项目全景

## 代码统计

nano-vLLM 全部核心代码约 **1,200 行**，分布在以下目录结构中：

```
nanovllm/
├── __init__.py              # 导出 LLM, SamplingParams
├── config.py                # 全局配置 dataclass
├── llm.py                   # LLM 入口类 (继承 LLMEngine)
├── sampling_params.py        # 采样参数
├── engine/
│   ├── llm_engine.py        # 推理引擎主循环
│   ├── scheduler.py         # 请求调度器
│   ├── block_manager.py     # KV Cache 块管理器
│   ├── sequence.py          # 序列状态定义
│   └── model_runner.py      # 模型运行器
├── models/
│   └── qwen3.py             # Qwen3 模型定义
├── layers/
│   ├── attention.py         # FlashAttention + Triton KV Cache 写入
│   ├── rotary_embedding.py  # RoPE 旋转位置编码
│   ├── linear.py            # 张量并行线性层
│   ├── layernorm.py         # RMSNorm (含残差融合)
│   ├── activation.py        # SiLU + Mul 门控
│   ├── sampler.py           # Gumbel-like 采样
│   └── embed_head.py        # 词嵌入 + LM Head
└── utils/
    ├── context.py            # 全局推理上下文
    └── loader.py             # safetensors 权重加载
```

## 核心抽象

只需记住 5 个核心抽象：

| 抽象 | 文件 | 职责 |
|------|------|------|
| `Sequence` | `engine/sequence.py` | 一个请求的完整生命周期 |
| `BlockManager` | `engine/block_manager.py` | KV Cache 的虚拟内存管理器 |
| `Scheduler` | `engine/scheduler.py` | 决定哪些请求可以运行、何时抢占 |
| `ModelRunner` | `engine/model_runner.py` | 模型推理执行 + KV Cache 物理存储 |
| `Context` | `utils/context.py` | 一次 forward pass 的元数据容器 |

## 一行调用展开

用户写下的这行代码：

```python
outputs = llm.generate(prompts, sampling_params)
```

在内部展开为以下循环（`llm_engine.py:73`）：

```
while not scheduler.is_finished():
    seqs, is_prefill = scheduler.schedule()    # 调度决策
    token_ids = model_runner.run(seqs, is_prefill)  # 模型前向
    scheduler.postprocess(seqs, token_ids, is_prefill)  # 状态更新
```

这就是整个推理框架的心跳——**schedule → run → postprocess** 循环。

## 三大核心问题

LLM 推理框架要解决的核心问题，本质上只有三个：

1. **显存怎么管？** → BlockManager（第2章）
2. **请求怎么排？** → Scheduler（第3章）
3. **模型怎么跑？** → ModelRunner + Attention（第4-5章）
