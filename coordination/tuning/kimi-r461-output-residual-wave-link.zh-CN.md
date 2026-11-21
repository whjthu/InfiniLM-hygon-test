# Kimi-K3 输出残差归约的双臂组件链接

在原 512×9×7168 BF16 组件输入上比较 full-tree 与 wave32。两臂共用一份 HOST 对象、权重入口、softmax 和最终 GEMM。每个动态库分别链接对应的一个 score 对象，保持现有两个组件 C ABI，并核对隐藏 score 符号由本库定义。实际链接使用原 .330 的编译、链接参数，.987 的编译器身份、.335 的依赖闭包，以及精确 CUDA 运行库。两臂的实际被动库依赖和导出表必须一致。

等原 score 编译实际成功终结，再由根协调者登记单次 4 CPU、8 GiB、0 DCU、300 秒的 CPU 链接。生成源使用已保存的原 13 文件包，编译账单 JSON 用 `columns` 和 `row` 保留独立查询原步骤的 `JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,AllocTRES`。已冻结的 score 编译收据、两个对象及其 M/MD 文件必须全部可核验。沿用原 Kimi CPU 容器的依赖路径，显式设置三套设备可见性掩码 `-1`，并用同一个实际启动 epoch 约束 Slurm、容器和命令。

```text
python -B tools/link_kimi_r461_output_residual_wave.py --generated-source <original-source-dir> --score-receipt-sha256 <original-sha256> --score-accounting <original-terminal-accounting.json>
```

终态产物位于 `/refactor/candidates/Kimi461-output-residual-wave-component-link-CPU-v1`，包括 `full-tree.so`、`wave32.so` 和 `CPU-result.json`。本机已验证实际渲染字节和链接命令的正负边界。目标 score 编译、动态库链接、GPU 数值、同卡性能和整模型验收待实测。

## 同卡组件对照

`tools/run_kimi_r461_output_residual_wave_pair.py pair` 接收已经独立核验的链接终态、两个库的收据，以及当次 PCI/KFD/PID 预检。三个文件均传实际 SHA-256。协调者绑定新作业的原启动 epoch 和唯一输出目录。设备端读取旧任务已经封存的 BF16 残差、原 norm 与 projection，读取和使用范围与原真实权重组件一致。原输入收据保持原字节。

六个独立进程全部调用 mode 2，顺序为 full-tree/wave32、wave32/full-tree、full-tree/wave32。每对按真实保存的完整 512×7168 输出计算差异。启动、调用、原设备内存采样分别记录。完整三对才取同步调用窗口的 wave32/full-tree 比值中位数。记录失败臂和缺失对。当前接口保存的是最终残差，内部 score 逐元素和整模型验收仍需各自证据。本入口尚未作为真实 Slurm 任务提交。
