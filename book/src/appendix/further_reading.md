# C. 扩展阅读

## 论文

1. **PagedAttention**: Efficient Memory Management for Large Language Model Serving
   - vLLM 的奠基论文，阐述了将操作系统虚拟内存分页思想引入 LLM 推理的方法
   - [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

2. **SGLang**: Efficient Execution of Structured Language Programs
   - 引入 RadixAttention（基数树注意力），优化 KV Cache 复用
   - [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)

3. **FlashAttention**: Fast and Memory-Efficient Exact Attention with IO-Awareness
   - IO 感知的精确注意力算法，是 PagedAttention 的算子基础
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

4. **FlashAttention-2**: Faster Attention with Better Parallelism and Work Partitioning
   - FlashAttention 的改进版本，提升 GPU 利用率
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

5. **Efficiently Scaling Transformer Inference**
   - 分析了 LLM 推理的 compute-bound 和 memory-bound 阶段
   - [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)

## 开源项目

| 项目 | 特点 |
|------|------|
| [vLLM](https://github.com/vllm-project/vllm) | 工业级 LLM 推理框架，PagedAttention 的原始实现 |
| [SGLang](https://github.com/sgl-project/sglang) | RadixAttention，结构化生成语言 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | 纯 C/C++ 实现，量化推理标杆 |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 本书分析对象，~1200 行极简实现 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | 高效注意力算子库 |

## 进阶方向

### 1. 从 nano-vLLM 到 vLLM

读完本书后，建议按以下顺序切入 vLLM 源码：

1. `vllm/core/block_manager.py` — 对比 BlockManager 的两层分离
2. `vllm/core/scheduler.py` — 理解 Swapped 队列和换出策略
3. `vllm/worker/model_runner.py` — 看多模态输入处理
4. `vllm/distributed/` — 理解 Ray 集群调度

### 2. 前缀缓存 vs RadixAttention

SGLang 的 RadixAttention 用基数树（Radix Tree）代替线性哈希链管理 KV Cache 复用：

- **优势**：基数树支持任意前缀匹配（不仅是完整块），可以复用部分块的 KV Cache
- **适用场景**：多轮对话、Agent 工作流中的大量重复 system prompt

### 3. Continous Batching vs Orca

nano-vLLM 和 vLLM 都采用 **continuous batching**（也叫 inflight batching）：只要有序列完成就立即用新序列填补空位，而非等整批完成。

对比传统 static batching：
- Static batching：等所有序列完成才处理下一批 → GPU 空闲
- Continuous batching：完成即替换 → GPU 利用率极高

### 4. Speculative Decoding

vLLM 支持投机解码：用小模型猜测多个 token，再用大模型并行验证。nano-vLLM 没有实现，但理解 PagedAttention 后，你会意识到 Speculative Decoding 的难点在于：如何高效管理 draft token 临时分配的 KV Cache 块——验证通过的保留，拒绝的释放。
