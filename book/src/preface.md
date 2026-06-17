# 前言

## 为什么要读 nano-vLLM？

vLLM 是当前工业级 LLM 推理框架的事实标准，但它的源码已经超过十万行，横跨分布式调度、多模型适配、自定义 CUDA 算子等众多领域。直接啃主干源码很容易迷失。

**nano-vLLM** 用约 1,200 行 Python 代码实现了 vLLM 的核心机制——PagedAttention、KV Cache 分页管理、前缀缓存、请求调度、CUDA Graph、张量并行——性能甚至可以与 vLLM 媲美。它是最理想的 LLM 推理框架学习载体。

## 本书的定位

这不是 API 文档，而是一份**深度解构笔记**。目标：

1. **逐文件、逐函数**剖析 nano-vLLM 源码，还原每一个设计决策背后的原理
2. 建立 nano-vLLM 代码与 vLLM 架构之间的**精确对位映射**，让你读完后再看 vLLM 源码有如从高维俯瞰
3. 突出**内存管理**和**请求调度**这两个 LLM 推理框架的灵魂

## 约定

- 所有源码引用格式：`文件路径:行号`，如 `nanovllm/engine/block_manager.py:58`
- 代码片段可能经过格式调整以提升可读性，但不改变语义
- 关键术语首次出现时会给出英文原文

## 你将获得

读完本书，你将深刻理解：

- **PagedAttention** 如何将操作系统虚拟内存的分页思想引入 GPU 显存管理
- **Block Manager** 如何实现逻辑块到物理块的映射、前缀缓存（Prefix Caching）、引用计数回收
- **Scheduler** 如何在 Prefill 和 Decode 之间切换、如何做抢占（Preemption）、如何实现 Chunked Prefill
- **CUDA Graph** 如何消除 decode 阶段的 kernel launch overhead
- **Tensor Parallelism** 在线性层和 Embedding 层上的精确实现方式
