# 第4章：模型执行引擎

ModelRunner 是 nano-vLLM 中与 GPU 直接交互的模块——它负责初始化模型、分配 KV Cache 显存、准备每步的输入数据、执行模型前向、以及 CUDA Graph 的捕获与重放。
