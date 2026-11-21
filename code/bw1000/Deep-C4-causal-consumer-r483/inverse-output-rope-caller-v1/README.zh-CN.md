# DeepSeek-v4 inverse RoPE 调用方

以归档的 `AttentionPrepare` 源码及头文件为唯一输入。脚本验证两份文件的 SHA256 后生成调用方，将 `inverse_output_policy.hpp` 放到同一目录。最终链接需要已独立核验的 H1/H2 编译对象。目标源码或头文件发生变化时，重新核对相同边界。

```text
python -B render_caller.py <original.cpp> <original.hpp> <derived.cpp>
python -B check_caller.py
```

进程首次执行 inverse output 时读取 `INFINILM_DEEP_INVERSE_OUTPUT_ROPE`。未设置与 `legacy` 使用原 copy 加 SDK RotaryEmbedding 入口。显式 `h1` 或 `h2` 只对连续 BF16 `[1,8,512]` 与有效 owner stream 生效。其他形状保留原入口。候选调用失败时抛出错误，由原 Work 生命周期负责同步。

两条路径共用原 I64 position 范围校验、YaRN 与 FP32 查表公式。候选用同一 owning stream 写入新分配的 BF16 输出，仅省略旧整张量预复制和 SDK 所需的逻辑行号张量。保留目标 CPU 编译、真实 provider 数值、当前固定模型质量及配对性能各项门槛。
