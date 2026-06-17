# B. vLLM vs nano-vLLM 对比表

## 架构对比

| 特性 | vLLM | nano-vLLM |
|------|------|-----------|
| 代码规模 | ~100K+ 行 | ~1,200 行 |
| 支持模型 | 50+ | 仅 Qwen3 |
| 调度策略 | Prefill/Decode/Swapped | Prefill/Decode |
| 内存管理 | BlockSpaceManager + BlockAllocator | BlockManager (合一) |
| 驱逐策略 | LRU + Swap + Recompute | LIFO + Recompute only |
| Chunked Prefill | 任意序列 | 仅第一个序列 |
| CUDA Graph | ✅ | ✅ |
| Tensor Parallel | Ray (RPC) | spawn + 共享内存 |
| 采样策略 | multinomial, greedy, top-p/k | Gumbel-Max |
| 前缀缓存 | ✅ (automatic) | ✅ (automatic) |
| 多模态 | ✅ | ❌ |
| Speculative Decoding | ✅ | ❌ |
| 前缀缓存哈希 | xxhash | xxhash |

## 性能对比 (Qwen3-0.6B, RTX 4070 Laptop)

| 指标 | vLLM | nano-vLLM |
|------|------|-----------|
| 输出 token 数 | 133,966 | 133,966 |
| 总耗时 | 98.37s | 93.41s |
| 吞吐量 | 1,361 tok/s | 1,434 tok/s |

nano-vLLM 在小模型场景下吞吐量反超 vLLM ≈ 5%，主要因为：
1. 更轻量的调度开销
2. Gumbel-Max 采样比 multinomial 更高效
3. `@torch.compile` 对小算子的融合优化
4. 没有 vLLM 的大量表计数/日志/统计开销
