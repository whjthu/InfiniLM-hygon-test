# 海光算力现状与阶段成果盘点

盘点时间为北京时间 2026-11-23 10:38。以下结论按实际 Slurm 记录、已归档的原运行证据和统一框架验收矩阵划分。

## 资源与接棒

| 指标 | 实查结果 |
|---|---|
| 当前持有 | `1103606` 正在运行，40 台、320 张 DCU。原 `1087656` 于 01:55:11 因 `NODE_FAIL` 提前结束并释放另外 320 张。 |
| 补申请 | `1194900` 申请 40 台、320 张，因优先级仍在排队。 |
| 已启动推理 | 10:32 实查 `1103606.1433`、`.1434`、`.1435` 均为 RUNNING，每步 8 张，共 24 张。三步均已进入权重加载。 |
| 备用设备准入 | 10:38 核对全部 40 台，3 台在跑。35 台共 280 张通过 PCI、KFD 和进程核验。另有 1 台设备观察权限不足，以及 1 台发现 KFD 持有者，各 8 张。 |
| 当前实测 busy | 未获同窗口数据。24 张活跃步骤对应当前分配的 7.5%。历史 320 卡采样五点共 1,600 个有效读数，均值 0.0%，测量时间为 11 月 23 日 06:47。 [原采样](../2026-11-22/training40-survivor320-original1392-five-driver-busy-points.json) |
| 训练等待储备 | 已就绪 0/5。十个待处置的候选或依赖任务仍绑定失效父作业的节点。准入工具已支持以存活 40 节点产出参数，历史原请求保留原身份，先迁移可开工范围。 [训练账本](../../../coordination/training/training80-backlog-v1.json) |

训练、恢复、原配方任务评分和真实模型诊断优先接棒。空余节点用短时、可独立收尾的真实热点算子对照补位。以实际 Slurm 步、驱动有效读数和独立清场确定交接；从结束到下一步启动的间隔计入闲置卡时。采用现有 5 分钟巡检。北京时间每日 18:00 后的首次成功巡检将日报追加到对应日期目录。

## 可按工程阶段交付的成果

- **统一训练框架与可复核验收入口。** 计算节点实际核对提交 `a68952cb2285d6593b665147249d14ec1f8cecbc` 的 393 个源码文件、统一入口和 17 项配方登记。这套源码、配方索引、A100 工程计时口径和验收检查可交付。正式第三阶段仍为 0/17。 [节点部署与 17 项注册](../2026-11-17/training-common-a68952c-live-17-case-registry.json) [验收矩阵](../../../coordination/training/training-case-acceptance-matrix-v1.zh-CN.md)
- **Qwen3-VL 原配方训练与检查点库存。** 八卡真实连续 25 步及原检查点已独立核验。领域训练恢复 `.544` 的五步确实执行，配对数值仍有漂移。可交付复现入口、阶段检查点身份和恢复异常定位。 [八卡原配方](../2026-11-14/training-qwen-1087656-220-native25-independent-terminal-and-checkpoint-inventory.json) [领域恢复对照](../2026-11-22/training-qwen-usgs-resume20-original544-five-updates-paired-drift.json)
- **Z-Image COCO20 单卡训练与恢复。** 原生连续 25 步与第 20 步新进程恢复的 5 步，在登记的更新边界逐张量精确对齐，独立清场完成。后续 `.1425` 实际更新 25 步，轨迹与原路径一致；被测有限值检查缩短约 35.7–37.4%，两个 Slurm 训练步骤均耗时 145 秒。Style18 另有 12 项开发集 CLIP 配对结果，尚缺独立盲测、正式 FID 和同配方 A100 效果。 [COCO20 恢复](../2026-11-22/training-zimage-COCO20-recovery-boundaries-v2-original1388-1390-exact-terminal.json) [25 步优化回归](training-zimage-original1425-grouped-finite-25-updates-terminal.json) [Style18 开发评分](../2026-11-22/training-zimage-style18-1149-openai-clip-dev12-terminal.json)
- **HunyuanVideo 1.5 单卡原生训练及分段耗时。** `.1401` 独立核验 25/25 步，Slurm 正常终结并通过清场。第 6、7 步反向主机区间均约 4.06–4.07 秒，Muon 更新约 0.206 秒。设备标记采用流区间，可能包含空闲。当前可交付 25 步功能证据和优化定位，A100 原始首步数值及正式三轮待验。 [实测分段](training-hunyuanvideo-original1401-25-updates-phase-profile-terminal.json)
- **VGGT TUM RGB-D 领域工程候选。** 九对 TUM 数据上已核验 25 步连续训练及五步精确恢复。可交付配方、检查点身份和恢复证据。原 CO3D 配方与留出集相机位姿效果单独待验。 [TUM 原执行与清场](../2026-11-22/training-vggt-nine-public-RGBD-original927-native25-resume20-terminal.json)
- **推理和算子阶段成果。** DeepSeek 三组旧 TP8 长输入全部完成，三组均未超过 15 tok/s；路由计数库本轮目标构建通过，988 项编译依赖和 68 个动态加载路径已核验，新三臂正在运行。GLM 有完整驻留的 prefill 配对与数值证据，尚无合格在线 decode。Kimi 有六臂设备配对，wave32 暂不晋级。历史 592 卡 BF16 组件对照有 1,184 个同卡配对，深度 21 对 15 的吞吐中位增益约 1.90%，低于 2% 晋级门槛；局部采样约 96.5–97.7%。这些是可交付的取证和选型依据。 [DeepSeek TP8](../2026-11-22/deep-rope-real-model-tp8-v2-original1376-1378-terminal.json) [计数库](Deep-rope-route-counter-original1429-target-build-terminal.json) [GLM 配对](../2026-11-22/GLM474-original1180-three-resident-pairs-terminal.json) [Kimi 六臂](../2026-11-22/Kimi461-wave-two-dso-original781-six-arm-independent-terminal.json) [592 卡算子](../2026-11-22/O01-depth15-21-592-card-original-terminal-and-clean-release.json)

## 17 项训练验收状态

各用例正式同配方海光三轮性能和正式留出集效果均待补证。下表记录已完成的工程证据和最接近的未过项。完整 A100 对照、原配方、量化门槛和各行证据可从验收矩阵复核。

| 用例 | 现有证据及待验项 |
|---|---|
| T01 Qwen3-VL SFT | 八卡连续更新已核，首步配对和恢复数值待收敛。 |
| T02 Gemma SFT | USGS72 工程恢复已执行五步，原配方首步和恢复待验。 |
| T03 InternVL SFT | 已核运行时和参考捕获，八 rank 首微批字段仍待比对。 |
| T04 LLaVA SFT | 已核 A100 捕获参考，海光完整权重和配对首步待验。 |
| T05 MiniCPM SFT | 原生输入与工程恢复有记录，恢复出现漂移，原未决句柄仍需对账。 |
| T06 SDXL SFT | 已实际更新，较新的原生 SDPA 对照全量参数更新误差约 19.4%，超过 5% 门槛。 |
| T06 SDXL DDPO | 原计时口径已核。固定 rollout、奖励、策略更新及恢复待验。 |
| T07 Z-Image SFT | COCO20 单卡 25+5 恢复精确通过。对应 A100 首步及正式性能、效果待补。 |
| T07 Z-Image Flow GRPO | 独立 rollout、策略更新与恢复待验。 |
| T08 HY-Motion Flow Matching LoRA | 现有记录覆盖原 60 帧口径，缺真实 128 帧与完整 201 维同输入配对。 |
| T08 HY-Motion MotionRFT | 原 A100 计时边界已核。真实奖励、策略更新与恢复待验。 |
| T09 HunyuanVideo SFT | 原生 25 步成功并获得两步分段耗时。历史 20→25 恢复的产物精确匹配，原步骤保留失败状态。A100 首步和正式性能待验。 |
| T09 HunyuanVideo DP2 | 已核 A100 最慢 rank 计时，海光双副本及双 rank 恢复待验。 |
| T09 HunyuanVideo SAGE-GRPO | scheduler 与恢复组件有证据，真实 actor/VAE 策略轮次及恢复待验。 |
| T10 TRELLIS.2 Shape Flow LoRA | 有原生首步证据，更新数值门槛未过。 |
| T11 Stable Fast 3D 材质头 | 原生 CPU 导入和 A100 首步参考已核，海光目标权重及材质首步待验。 |
| T12 VGGT 相机头 | TUM 工程训练与恢复可交付，原 CO3D 数值、恢复和位姿指标待验。 |

权重、原始样本、完整检查点和大体积日志保留在实验存储。报告通过固定运行句柄、提交、原件路径及 SHA-256 关联证据。未决、失败和未测项保留在各自分母内。 [证据索引](training-cluster-delivery-review-evidence.json)
