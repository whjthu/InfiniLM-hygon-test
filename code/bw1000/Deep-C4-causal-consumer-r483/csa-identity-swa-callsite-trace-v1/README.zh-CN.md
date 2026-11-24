# DeepSeek CSA SWA 提交分支观测

生成器使用原 TU07 候选的固定源码，只在 `CsaAttention::forward` 的实际提交位置添加观测。分别记录使用原 KV 的分支、真实 `IndexSelect` 提交成功、原 cache store 返回以及原流同步返回。记录绑定实际 `Runtime` 指针和同一线程。输出包含一次 forward 的序号、请求数量、提交行以及四个观测计数。异常路径不会把未完成的 store 或同步记为成功。

现有两状态行、32768 上下文计划选取五个证据步骤。`trace_io.py` 验证实际记录，按原请求检查跨行混合、窗口裁剪和最大 past 等几何，并为两臂分别保存独立的 `NNN-commit-route.json`。证据消费端通过 `verify_trace()` 重新读原记录并核对捕获 SHA 和实际 `IndexSelect` 计数。

本机 C++ 记录器和 Python 证据测试已通过。当前仅生成目标源文件，尚未构建或执行新 TU。正式性能仍使用已合格的无观测版本和独立计时。
