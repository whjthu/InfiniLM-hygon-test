# VGGT CO3D 留出计分输入与验收证据

`tools/vggt_co3d_effect.py` 汇总 T12 的 A100、海光训练后和海光训练前角误差，输出 AUC@5、AUC@15、AUC@30 与框架验收器可读取的指标行。先为每个图像对取旋转、平移角误差的较大值。组内按 VGGT 评估实现的一度直方图和累计分布求 AUC，再对各类别等权平均。每道阈值的最后一个 bin 包含其右边界。输出使用 0–1 的比例。[官方固定实现](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py)

计划 JSON 固定 `case=T12.camera-head`、`framework_commit`、`dataset_SHA256`、`scorer_SHA256`、`split=test`、`pose_evaluator_commit=4b8be14b574b58c91ecd699122daf3d8004901d4` 以及 `policy=vggt-co3d-frozen-categories-and-explicit-zero-credit-failures-v1`。`sequences` 列出每个 `category`、`sequence` 和有序 `frame_ids`。从同一序列展开全部图像对，`sample_id` 由 `category/sequence/第一帧/第二帧` 组成。`--scorer-sha256` 返回本计分源码 SHA-256。

预测清单的 `evaluation_plan_SHA256` 绑定计划，`arms` 分别记录 `A100`、`Hygon`、`Hygon_before` 的 `path`、预测 JSONL 的 `SHA256` 和 `checkpoint_SHA256`。每份 JSONL 首行重复 `arm`、`checkpoint_SHA256`、`evaluation_plan_SHA256`、`pose_evaluator_commit`。随后每行包括 `sample_id` 和 `status`。`scored` 提供旋转误差 `rotation_deg` 与官方平移方向歧义约定下的 `translation_deg`。`failed` 提供 `reason` 并计零。所有组别使用同一个冻结分母，缺失或重复图像对会使本次计分失败。原版评估会跳过部分输入，本工具按项目的失败口径计入计划中的失败项。

```text
python -B tools/vggt_co3d_effect.py --ledger-root LEDGER --plan PLAN.json --plan-sha256 PLAN_SHA --predictions PREDICTIONS.json --predictions-sha256 PREDICTIONS_SHA --output LEDGER/NEW_OUTPUT
```

`co3d-metric-rows.json` 可作为 `holdout.metrics`。三个指标使用同一 `dataset_SHA256` 和 `scorer_SHA256`。计分回放需要在有权访问原件的位置读取冻结清单与每组角误差，用同一脚本重算。正式验收还需核对 CO3D 原测试标注、媒体与训练集隔离，以及三组从实际检查点运行同一姿态评估流程的原始证据。沿用原项目 CO3D 数据范围时，公开评估的全量类别得分不充当项目 A100 基线。[官方 CO3D 评估说明](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/README.md)

本机通过 4 项工具回归和 25 项现有效果及性能验证器回归。另按官方 NumPy 直方图结构交叉核对 1,215 组含边界和随机值的分布，最大绝对差为 `2.22e-16`。本轮输入为合成角误差，实际测试集的三组模型预测和效果结论仍缺原件。
