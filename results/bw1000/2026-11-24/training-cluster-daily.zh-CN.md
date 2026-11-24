# 海光训练、推理与算力日报

日期为 2026 年 11 月 24 日。时间均为北京时间。本次记录截至 18:05:58。

## 算力与接续

17:19:16 至 17:19:26，同步覆盖 80 台、640 张 DCU 的五点驱动采样有效样本为 3,200，平均忙度 94.15%。604 张卡五点均严格大于 80%，同一批 32 张卡五点均为零。采样前后均有 608 张卡处于编号 RUNNING GPU 步骤。该窗口覆盖约 8 秒，全天连续忙度和闲置卡时未取得。 [同步采样及独立复算](training80-1719-synchronized-640-busy-independent.json)

32 张缺口涉及 f12r2n12、f12r3n03、f13r4n06、f13r4n09。其中三台旧步骤账务与控制器冲突，另一台上次物理实查由其他账号持有 KFD，新诊断未取得回执。普通接续 70 台预排九批，首批 17:39 的现场核查均仍运行。逐台核对旧步骤结束和设备释放后接续，保底与有效业务分报。 [缺口与接续范围](training80-1713-running-coverage-and-standby-renewal-plan.json)

晚间滚动接续已有 24 台的计算侧桥接回执，申请 192 卡。前三台的实际后继 1194900.793/.794/.795 已实查 RUNNING，尚未核到逐卡计算计数增长。T07 的旧 GPU 步骤及零卡控制均已正常结束，后继仍待确认。Pack16 的零卡依赖检查 1103606.2569 已实际执行、子进程返回 1，独立 Slurm 终态仍待核对。以上不计入新的全池利用率。 [晚间现场证据](training80-1805-ordinary-wave-startup-and-pack16-probe.json)

## 今日工程产出

| 项目 | 可复核结果及验收缺项 |
|---|---|
| Z-Image-Turbo | 统一候选 55cfdaf7 的海光三轮中位 1.25803 updates/s，同口径 A100 1.09524，比例 114.86%。另一个统一候选实测连续 25 步和从第 20 步恢复的 5 步一致。仍缺 A100 全量同输入数值和规范留出集效果。 [统一候选](training-common-55cfd-zimage-window-and-DP2-backend-lineage.json) |
| HunyuanVideo DP2 | 原 A100 两个 rank 的首步 flow 指纹各自在三轮间一致。海光两个 rank 与 A100 均不匹配。已核对可导入的捕获和回放候选 9338c7eb，尚无 A100 捕获或海光同输入回放。 [首步证据](training-hunyuanvideo-DP2-original-A100-and-Hygon-first-flow-mismatch.json) |
| MiniCPM | 原 1103606.2412 失败。事后复算 1,488 对掩码完全一致，但 5 个 rank 的多次 loss 仍变化。原失败和恢复门槛仍保留。 [原终态及复算](training-minicpm-native-mask-original2412-terminal-and-posthoc.json) |
| InternVL | 17:34 八卡保底仍运行。训练原请求状态未确认。自动审批拦截了会永久阻止原请求启动的封存，实际未执行。 [只读对账](training-internvl-original-v2-1734-standby-live-and-fence-review-gate.json) |
| TRELLIS.2 | 四张公开模型输入通过原生预处理核验。训练适配器生成入口已完成零卡 Torch/SafeTensors 载入测试。仍缺配对的实际模型生成、有效更新和几何评分。 [输入与组件清单](training-trellis-heldout4-native-preparation-and-target-inventory.json) · [零卡实测](training-trellis-shape-adapter-original2187-CPU-terminal.json) |
| 共享算子 | Z-Image 的 288×3840×3840 BF16 算子候选完成本机编译及四项调用验证。实测核对 14,745,600 个布局元素。是否能在原训练中连续复用同一 RHS 16 次以及目标卡净收益均待实测。 [候选及本机证据](O01-zimage-square-rhs-pack16-frozen-source-local-review.json) |
| InfiniLM | DeepSeek-v4 单个 HOST 编译单元在 1194900.703 完成，新增 3 个观测入口，无模型调用或 GPU 编译单元。GLM-5.3、DeepSeek-v4、Kimi-K3 的逐用户严格大于 15 tok/s 均待真实在线验收。 [DeepSeek 编译终态](Deep-CSA-identity-SWA-callsite-trace-original703-CPU-terminal.json) |

统一入口已核对 17 项训练用例，阶段三通过 **0/17**。功能、数值、恢复、同配方性能及逐项留出集证据见完整矩阵。 [训练验收矩阵](../../../coordination/training/training-case-acceptance-matrix-v1.zh-CN.md)
