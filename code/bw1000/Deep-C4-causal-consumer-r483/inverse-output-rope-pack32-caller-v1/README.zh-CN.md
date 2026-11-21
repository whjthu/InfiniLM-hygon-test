# DeepSeek-v4 inverse RoPE 的 pack32 调用方

在已有 `AttentionPrepare` 调用方中增加 `INFINILM_DEEP_INVERSE_OUTPUT_ROPE=h1-pack32`。候选沿用 H1 的输入形状、owner stream、YaRN 表、缓存生命周期和同步边界。仅连续 BF16 `[1,8,512]` 且有 stream 时进入新提供者。指针对齐的 16 位回退由同一个 pack32 提供者处理。未设置时走原调用路径，`h1` 和 `h2` 仍各自选用原提供者。

```text
python -I -S -B render.py <new-local-source-dir>
python -I -S -B check_caller.py
```

渲染器核对归档原文件和调用方的已发布身份，生成修改后的 `deepseek_v4_attention_prepare.cpp`、`inverse_output_policy.hpp`，以及绑定候选源码的 `CALLER_BINDING.json`。最终目标链接仍需要独立验证原 H1/H2 和新 pack32 两个 gfx936 对象及其导出，冻结原 Native 依赖和完整编译闭包。原请求的执行状态仍按原记录处理。

本机测试编译了渲染输出中的真实调用方法，注入可记录参数的 CPU 测试提供者，覆盖默认及显式策略、形状和 stream 回退、错误传播和归档源码漂移。目标翻译单元、SDK、GPU 数值、性能与完整模型服务均待实测。
