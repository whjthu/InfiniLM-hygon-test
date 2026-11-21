# Kimi r461 输出残差内部 score 诊断

`tools/prepare_kimi_r461_output_residual_score_capture.py` 从原双臂清单生成单独的诊断源，校验原清单后只替换两臂相同的 `component.cpp`。评分翻译单元、归约头、原生残差计算、输入及权重均沿用原双臂包。诊断清单 SHA-256 为 `3755a70a0603e32c38b36b909c3b28672f34ea3b0f3cb55edde601c024fb2967`。

CPU linker 的 `--score-capture` 模式使用独立目标目录，复用原两份 score 目标文件及其终态、MD 和供应库证据。链接后核对两个 DSO 的 `kimi_output_residual_scores_probe_r461` 导出。原组件入口继续使用它原来的 ABI。诊断入口只接受 mode 2 及 4,608 个 BF16 元素的输出缓冲。

GPU worker 的 `pair --score-capture` 模式绑定诊断清单、两组 DSO、子进程编号和设备证据。原生层完成相同 dispatch 同步后，从活跃的 `[512,9,1]` BF16 score Tensor 回读 9,216 字节。每臂检查 host 缓冲两侧哨兵、完整性与有限值。每对分别比较全部内部 score 和最终 BF16 残差，保留诊断的绝对计时和三对失败分母。回读及审计计入 post-dispatch，诊断不产生两臂速度比。原始 BF16 保留在私有结果目录。

本机验证覆盖原清单和诊断清单隔离、两组 CUDA 源字节复用、ABI 和导出约束、容量及部分回写错误、完整 score 逐字比较。CUDA/DTK 下目标 HOST 编译、实际链路与 GPU 结果仍待取证。该诊断还需同一设备的真实可用性、精确工作范围及原预算准入。
