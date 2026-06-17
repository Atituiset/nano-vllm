# 1.3 与 vLLM 的对位映射

理解 nano-vLLM 的另一个价值：它提供了 vLLM 核心模块的**最小对应物**。下表精确映射了两者之间的等价关系：

## 模块映射

| nano-vLLM | vLLM | 说明 |
|-----------|------|------|
| `LLMEngine` | `LLMEngine` | 完全同构，都是 schedule→run→postprocess 循环 |
| `Scheduler` | `Scheduler` | 核心逻辑一致，vLLM 多了 swapping（交换到 CPU） |
| `BlockManager` | `BlockSpaceManager` + `BlockAllocator` | nano 合并为一个类；vLLM 拆分为逻辑块管理 + 物理块分配器两个层次 |
| `Sequence` | `Sequence` + `SequenceGroup` | vLLM 区分 Sequence（单次生成）和 SequenceGroup（beam search 等），nano 只有 Sequence |
| `ModelRunner` | `ModelRunner` | 结构相同；vLLM 多了多模态输入处理 |
| `Context` | `InputMetadata` / `AttentionMetadata` | 等价：一次 forward 的元数据 |
| `store_kvcache_kernel` | `copy_cache_op` / `reshape_and_cache` | 等价：写 KV Cache 的 GPU kernel |
| `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | 模型定义完全对齐 |

## vLLM 多了什么

| 特性 | vLLM | nano |
|------|------|------|
| Swapping（KV Cache 换出到 CPU） | ✅ | ❌ 仅抢占 |
| Beam Search / Parallel Sampling | ✅ | ❌ 单序列生成 |
| 多模态输入 | ✅ | ❌ |
| 多模型架构 | ✅ 50+ | ❌ 仅 Qwen3 |
| 分布式调度（Ray） | ✅ | ❌ 单机 spawn |
| Speculative Decoding | ✅ | ❌ |
| LogProb / PromptLogProb | ✅ | ❌ |

## nano 精简了什么

理解精简掉了什么，才能看清**本质**：

1. **没有 Swapping**：nano 用抢占（释放块、重计算）替代换出（拷到 CPU、再拷回 GPU），简化了调度状态机
2. **没有 SequenceGroup**：vLLM 用 SequenceGroup 支持一次请求多条并行生成路径（如 beam search），nano 只支持 1:1
3. **没有 GPU 端 Block Table 更新**：nano 在 Host 端维护 Block Table，每步从 CPU 拷贝到 GPU
4. **没有 Sliding Window**：vLLM 支持滑动窗口注意力，nano 不做截断

这些精简让核心路径一目了然。
