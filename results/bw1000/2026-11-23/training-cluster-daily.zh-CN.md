# 海光训练、推理与算力日报

日期为 2026 年 11 月 23 日，本文时间均为北京时间。

## 算力与补位

两份长时父作业 `1103606` 和 `1194900` 各持有 40 台、320 张卡，合计 80 台、640 张卡。21:02:54，一批保底步骤集中到期后还剩 200 张卡有运行中的 GPU 步骤。21:03:52 的独立复查为 192 张。专职智能体逐台重新核对归属和设备后，分三批补回 50 台、400 张卡。21:12:39，Slurm 再次确认 **592/640 张卡**有运行中的 GPU 步骤，覆盖率 **92.5%**。

三批新补的 400 张卡均在相同设备、相同 Slurm 步骤的两次读取间出现 BF16 调用计数增长。按这 400 张卡的异步驱动读数加权，busy 均值为 **95.80%**。21:15:47 的复查仍有 592 张卡运行。全池同步 busy 和连续闲置卡时尚缺完整窗口。

剩余 48 张卡涉及 6 台。4 台旧步骤的控制器与记账状态冲突，1 台运行零卡控制步骤，1 台有未释放的 KFD 持有者。巡检继续核对原句柄和下一批实际到期时间。[本轮全池补位证据](training80-standby-expiry-wave-2103-and-refill.json)

## 今日实测

| 工作项 | 已核验证据 | 验收状态 |
|---|---|---|
| Z-Image-Turbo 基础训练 | 统一注册入口完成连续 25 步及从第 20 步检查点恢复的 5 步。恢复输入、loss、476 个 LoRA 张量及末步检查点一致。单卡步骤 1103606.1787 匹配 420/420 条实际 BF16 GEMM 操作数关联，覆盖三个形状和六种布局。 | A100 全量首步对照、海光三轮完整窗口、留出集仍待验。 |
| HunyuanVideo DP2 | 双进程原配方已运行。海光单轮 0.141178 updates/s，为原 A100 三轮中位数的 71.04%。 | 待补 A100 同输入首步数值、正式三轮和留出集。 |
| VGGT 领域候选 | 两场景开发集 24 窗已审计。候选 AUC@30 为 0.19934，基线为 0.39788。两边各有 2 个失败配对，保留全部分母。 | 已终止本增广路线。正式 CO3D 仍单独验收。 |
| GLM-5.3 | 八卡组件小试的两个顺序中位数为 1.04734 和 0.98849，未同时达到预注册的 +3% 门槛。 | 保留原路径，该候选不提高上线级别。 |
| DeepSeek-v4 | 24 卡路线诊断的三臂输出 ID 一致，逐 owner 的 decode 为 2.351、2.366、2.381 tok/s。臂之间用了不同节点。 | 在线服务和逐用户 15 tok/s 仍未验收。 |
| Kimi-K3 | 最近已归档的硬件组件证据为 11 月 22 日六臂试验。Wave32 对原路径的完整接口耗时比例约 1.000。 | 未采用该改动，整模型服务目标仍待验。 |

统一训练入口已核验可枚举 17 项用例。阶段三通过数为 **0/17**。完整配方、数值、恢复、吞吐及效果按用例记录在[验收矩阵](../../../coordination/training/training-case-acceptance-matrix-v1.json)。Z-Image 的同口径计时入口已在目标端核验，从第 6 步取数前到第 25 步检查点落盘，正式 GPU 三轮尚未执行。

## 调优与下一次验收

Z-Image 首个试验形状为 `(M,N,K)=(288,3840,3840)`。真实右操作数的连续及转置布局各出现 120 次。对照源码已部署到目标实验目录并核对原布局，GPU 对照尚未执行。配对只改变选用的 GEMM 路径，两个布局各自保留完整配对及数值证据。只有实际执行路径发生变化时，才解释计时差。训练数值、留出效果与组件吞吐分别验收。

## 原件索引

- [Z-Image 连续训练与恢复](training-zimage-registered-1759-1761-1763-recovery-exact.json)
- [Z-Image 实际 GEMM 布局及节点恢复](training-zimage-registered-1787-actual-gemm-operands-terminal.json)
- [HunyuanVideo DP2](training-hunyuanvideo-dp2-live-original1612-terminal-recovery-and-standby.json)
- [VGGT 开发集](training-vggt-two-scene-dev24-v2-original1506-terminal.json)
- [GLM 八卡组件小试](GLM-headshare-pilot-original1456-eight-card-terminal.json)
- [DeepSeek 路线诊断](Deep-rope-tp8-route-attribution-original1433-1435-terminal.json)
- [Kimi 六臂组件试验](../2026-11-22/Kimi461-wave-two-dso-original781-six-arm-independent-terminal.json)
