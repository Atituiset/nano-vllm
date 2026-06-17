# 第5章：注意力层与 KV Cache 读写

Attention 层是整个推理框架中实现 PagedAttention 的核心——它负责：1) 将新计算的 K/V 写入 KV Cache；2) 从 KV Cache 中读取历史 K/V 执行注意力计算。
