# 长 prefill 窗口之间推进已驻留短请求

这是下一项默认关闭的源码候选，GPU_GO=false。当前运行的 r355 源码和作业未修改。r355 两个预热对的真实 emitter 时间线已显示：短请求首 token 后等待整个长 prefill，最大 ITL 约253/249秒；预热差异不能作为性能配对收益。

候选复用当前038d Native、C2/TP8/PP4、两个完整请求和原 W4 执行器。先完成短请求 prefill。在长请求每个完整 W4 组（最后不足四个 forward 也算完整末组）结束、原取消投票通过后，为仍活跃的短请求执行一次单 row decode 和真实反馈，再继续长 prefill。组内不插入 WORLD 投票、decode 或反馈。原输入、独立 past、输出预算、EOS 撤退、上下文上限保持；不补新请求，不重置正在使用的缓存。失败向外传播，由原 owner/controller 清理，不能用前端账本重播已部分完成的 Native 工作。

两个 arm 都固定使用 r355 的延迟反馈 decode。因此本卡只比较串行 prefill 与窗口间推进，不把当前355的 decode policy 同时当作变化变量。末组长请求 logits 保持独立所有权，短请求 forward 之后才采样；真实设备上这种交错仍待验证。

`owner_r290.py` 是对当前355 owner 的最小覆盖源，`prefill_interleave.py` 是它直接调用的消费者。三个依赖文件与355逐字节相同，manifest 记录原43成员包、原 owner 及覆盖文件 SHA。当前尚无新的完整部署包、控制器认领或 GPU 运行。

本地检查调用真实新消费者、原 W4 和 decode 执行器，Native 是记录替身。验证完整 prompt/KV/逐 row 输出状态、仅组边界插入、末组 logits 独立、EOS/预算/非法 context/取消和 Native 错误停止。实际 owner 覆盖点做 AST 检查；这不代表整个 owner 入口、Native 数值或设备通过。

后续先消费355正式四对及清理，由周期审核决定是否接本卡；若保留，再闭合完整 owner/controller 的 Linux CPU 入口和资源认领，才考虑一次同驻留的对照。记录 admission→first/last、最大及P95/P99 ITL、每用户速度、完整请求完成和32卡总吞吐。所有 prefill 等待都算入实际 token 时间线；buffered HTTP 完成时间不能冒充增量 TTFT/ITL。本卡未证明严格15 tok/s、开放到达、生产质量或任何模型收益。
