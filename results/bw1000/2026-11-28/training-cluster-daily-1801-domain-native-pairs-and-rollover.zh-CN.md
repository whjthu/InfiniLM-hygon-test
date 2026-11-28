# 海光训练、推理与算力日报 · 2026-11-28

## 交付与验收

| 指标 | 本日已验总数 | 主要缺口 |
| --- | ---: | --- |
| 统一框架正式训练用例 | 0/17 | 各原配方仍需补足同输入数值、恢复、三轮 A100 性能和原留出集 |
| 真实在线逐用户推理 | 0/3 | 三个模型均缺按客户端逐请求结算的 >15 tok/s 验收 |
| 正式领域模型底座覆盖 | 0/12 | 12 个主方向已明确，现有候选缺独立效果或完整恢复证据 |
| 正式领域模型交付 | 0/50 | 继续先做首批一底座一模型 |
| 可立即提交的真实训练储备 | 0/5 | 三次保底准入均实查为零。新发布的两项是检查点评测，不能计作有优化器更新的训练储备 |

## 今日业务证据

- **SDXL 商品棚拍。** 连续训练 25 次，并从 CP20 在新进程恢复 5 次。第 21–25 步的样本、loss、裁剪前梯度范数与最终状态一致。固定三个未参与训练的商品家族、六对种子，生成步骤 `1194900.2733` 在首图失败，0/12 出图。原错误摘要匹配 BF16 输入与 FP32 bias 冲突。训练和恢复已取得实证，领域效果待测。 [25 步及恢复](specialized-sdxl-2675-native-full25-cp20-exact-recovery-and-checkpoints.json) · [生成终态](specialized-sdxl-cp25-pairs-2733-native-vae-dtype-error-and-clean.json)
- **LLaVA 场景物体清点。** CP25 已从新进程复载，504 个 LoRA 张量逐值一致。GPU 步骤 `1103606.5203` 定位了基座参数误用装配后计数的问题。修复后的 `1103606.5207` 通过 698 个 BF16 基座参数、零 LoRA、一次原权重赋值及零缺失、意外或形状不匹配的键，首个原生生成遇到对 `None` 取下标的异常，保留 1 错误、31 缺失。两轮均独立清场，未评分。 [计数根因](specialized-llava-clevr-val16-5203-native-base-count-cause-and-clean-post.json) · [首个生成终态](specialized-llava-clevr-val16-5207-checkpoint-load-pass-first-generation-none.json)
- **Gemma 地震图表读数。** 最近的独立 96 题确认测得 CP25 答对 52 题，底座答对 40 题，提升 12.5 个百分点。目标端已核验原通用 TextVQA 验证集的 3,087/3,166 张图，余 79 张待对账，另需补恢复一致性和原能力保持。 [领域评分](../2026-11-27/training-gemma-usgs-confirmatory96-original3682-first-offline-score.json) · [TextVQA 目标读回](training-gemma-textvqa-val5000-target-72-batches-3087-images.json)

## 算力

北京时间 **17:59:29** 的 Slurm 核验确认仍持有 **80 台、640 卡**。一轮保底作业到期后，运行编号覆盖降至 456 卡。三批重新准入 136 卡，独立读回恢复到 **592/640 卡**，该快照的 592 卡为保底，48 卡没有运行中的 GPU 步骤。48 卡分布在六台节点，其中三台存在历史 Slurm 步骤冲突，另三台分别有设备持有、控制用途或历史 CPU 步骤条件。快照不证明驱动忙度。最近一段全 640 卡完整小时的实测平均忙度为 **80.14%**，今日晚间尚无新完整小时，也无可核的业务、保底、空闲及受阻卡时。 [补位结果](training80-1759-standby-rollover-456-to-592-numbered.json) · [完整小时原证据](../2026-11-27/training80-d895426c-verified-complete-hour-driver-busy-terminal.json)

## 下一批工作

先用零卡资格验证定点解决 SDXL 解码精度与 LLaVA 首个生成异常，再分别启动原配对验证。12 底座的训练储备沿着已有原生输入继续推进，首批关注 Qwen 街景 OCR、SDXL 棚拍、SF3D 材质、HY-Motion 转向动作与 TRELLIS.2 硬表面。已有检查点的模型继续按冻结的独立留出集验收。 [首批 12 底座执行单](../../../coordination/training/specialized-posttraining-first-12-delivery-sprint-20261128.zh-CN.md)
