# A. 核心数据流图

## Prefill 数据流

```
用户 prompt
    │
    ▼ tokenize
[prompt_token_ids]
    │
    ▼ Sequence.__init__
Sequence(num_tokens=N, status=WAITING, num_cached_tokens=0)
    │
    ▼ scheduler.add()
waiting 队列
    │
    ▼ scheduler.schedule() [prefill 分支]
    │
    ├─ can_allocate(seq) → num_cached_blocks
    │   │
    │   └─ 逐块计算 hash → 查 hash_to_block_id → 命中?
    │
    ├─ allocate(seq, num_cached_blocks)
    │   │
    │   ├─ 缓存块: block.ref_count += 1
    │   └─ 新块: _allocate_block() → block_table 追加
    │
    ├─ seq.num_scheduled_tokens = min(num_tokens, remaining)
    │
    ▼ model_runner.prepare_prefill()
    │
    ├─ input_ids = seq[cached:scheduled]   (跳过缓存 token)
    ├─ positions = range(cached, end)
    ├─ cu_seqlens_q/k = 累积序列长度
    ├─ slot_mapping = block_id × block_size + offset
    └─ block_tables (仅前缀缓存时)
    │
    ▼ Context.set_context(is_prefill=True, ...)
    │
    ▼ model(input_ids, positions)
    │   │
    │   ├─ Embedding
    │   ├─ for layer in layers:
    │   │   ├─ RMSNorm (含残差融合)
    │   │   ├─ QKV Projection
    │   │   ├─ QK Norm (Qwen3 特有)
    │   │   ├─ RoPE
    │   │   ├─ store_kvcache(k, v, k_cache, v_cache, slot_mapping)  ← Triton
    │   │   ├─ flash_attn_varlen_func(q, k, v, cu_seqlens, ...)
    │   │   ├─ O Projection (RowParallel, all_reduce)
    │   │   ├─ RMSNorm (含残差融合)
    │   │   └─ MLP (gate_up → silu_mul → down)
    │   └─ Final RMSNorm
    │
    ▼ LM Head
    │   └─ 取每个序列最后一个 token → logits
    │
    ▼ sampler(logits, temperatures) → token_ids
    │
    ▼ scheduler.postprocess()
    │
    ├─ hash_blocks(seq) → 更新已满块的哈希
    ├─ seq.num_cached_tokens += num_scheduled_tokens
    ├─ seq.append_token(token_id)
    └─ if finished → deallocate(seq)
```

## Decode 数据流

```
running 队列中的序列 (每序列 last_token 已就绪)
    │
    ▼ scheduler.schedule() [decode 分支]
    │
    ├─ can_append(seq) → 检查空闲块
    │   └─ 若不足 → preempt(running.pop()) → 释放块
    ├─ may_append(seq) → 若需要新块则分配
    └─ seq.num_scheduled_tokens = 1
    │
    ▼ model_runner.prepare_decode()
    │
    ├─ input_ids = [seq.last_token for seq in seqs]
    ├─ positions = [len(seq)-1 for seq in seqs]
    ├─ slot_mapping = [block_table[-1] × block_size + offset]
    ├─ context_lens = [len(seq) for seq in seqs]
    └─ block_tables = 填充为等长
    │
    ▼ Context.set_context(is_prefill=False, ...)
    │
    ▼ CUDA Graph 重放 (或 eager 执行)
    │   │
    │   ├─ Embedding (单 token)
    │   ├─ for layer in layers:
    │   │   ├─ RMSNorm
    │   │   ├─ QKV Projection (单 token, 计算量极小)
    │   │   ├─ RoPE
    │   │   ├─ store_kvcache (写入 1 个 slot)
    │   │   ├─ flash_attn_with_kvcache(q[1], k_cache, v_cache, context_lens, block_tables)
    │   │   ├─ O Projection + all_reduce
    │   │   └─ MLP
    │   └─ Final RMSNorm
    │
    ▼ LM Head → logits → sampler → token_ids
    │
    ▼ scheduler.postprocess()
    │
    ├─ hash_blocks (通常不更新，因为块未满)
    ├─ append_token(token_id)
    └─ if finished → deallocate
```
