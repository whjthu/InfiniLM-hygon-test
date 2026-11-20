# VGGT CO3D 留出计分输入与验收证据

`tools/vggt_co3d_effect.py` 汇总 T12 的 A100、海光训练后和海光训练前角误差，输出 AUC@5、AUC@15、AUC@30 与框架验收器可读取的指标行。先为每个图像对取旋转、平移角误差的较大值。组内按 VGGT 评估实现的一度直方图和累计分布求 AUC，再对各类别等权平均。每道阈值的最后一个 bin 包含其右边界。输出使用 0–1 的比例。[官方固定实现](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py)

计划 JSON 固定 `case=T12.camera-head`、`framework_commit`、`dataset_SHA256`、`scorer_SHA256`、`split=test`、`pose_evaluator_commit=4b8be14b574b58c91ecd699122daf3d8004901d4` 以及 `policy=vggt-co3d-frozen-categories-and-explicit-zero-credit-failures-v1`。`sequences` 列出每个 `category`、`sequence` 和有序 `frame_ids`。从同一序列展开全部图像对，`sample_id` 由 `category/sequence/第一帧/第二帧` 组成。`--scorer-sha256` 返回本计分源码 SHA-256。

预测清单的 `evaluation_plan_SHA256` 绑定计划，`arms` 分别记录 `A100`、`Hygon`、`Hygon_before` 的 `path`、预测 JSONL 的 `SHA256` 和 `checkpoint_SHA256`。每份 JSONL 首行重复 `arm`、`checkpoint_SHA256`、`evaluation_plan_SHA256`、`pose_evaluator_commit`。随后每行包括 `sample_id` 和 `status`。`scored` 提供旋转误差 `rotation_deg` 与官方平移方向歧义约定下的 `translation_deg`。`failed` 提供 `reason` 并计零。所有组别使用同一个冻结分母，缺失或重复图像对会使本次计分失败。原版评估会跳过部分输入，本工具按项目的失败口径计入计划中的失败项。

```text
python -B tools/vggt_co3d_effect.py --ledger-root LEDGER --plan PLAN.json --plan-sha256 PLAN_SHA --predictions PREDICTIONS.json --predictions-sha256 PREDICTIONS_SHA --output LEDGER/NEW_OUTPUT
```

`co3d-metric-rows.json` 可作为 `holdout.metrics`。三个指标使用同一 `dataset_SHA256` 和 `scorer_SHA256`。计分回放需要在有权访问原件的位置读取冻结清单与每组角误差，用同一脚本重算。正式验收还需核对 CO3D 原测试标注、媒体与训练集隔离，以及三组从实际检查点运行同一姿态评估流程的原始证据。沿用原项目 CO3D 数据范围时，公开评估的全量类别得分不充当项目 A100 基线。[官方 CO3D 评估说明](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/README.md)

本机通过 4 项工具回归和 25 项现有效果及性能验证器回归。另按官方 NumPy 直方图结构交叉核对 1,215 组含边界和随机值的分布，最大绝对差为 `2.22e-16`。本轮输入为合成角误差，实际测试集的三组模型预测和效果结论仍缺原件。

## 从已保存的相机外参生成逐对角误差

`tools/vggt_co3d_pose_receipts.py` 读取同一计划下的真值及三组推理外参。每份 JSON 记录 `coordinates=opencv-world-to-camera-3x4`、`evaluation_plan_SHA256` 和 `sequences`。每个序列包含 `category`、`sequence` 与按计划排序的 `frames`，每帧提供 `frame_id` 及 `extrinsic`。三份预测还记录 `arm`、`checkpoint_SHA256` 和 `pose_evaluator_commit`，逐帧标记 `status=scored` 或 `status=failed`。失败帧提供 `reason` 并省略 `extrinsic`。模型外参清单使用 `evaluation_plan_SHA256` 与 `arms`，每组都给出外参文件 `path`、`SHA256` 和 `checkpoint_SHA256`。

转换器在 CPU 上计算 `E_i · inverse(E_j)`，输出相对旋转角误差及允许平移符号歧义的方向误差。零平移在方向统计中记为 90°，显式失败帧对应的全部图像对保留为失败。清单不全或外参不是有效刚体变换时停止计分。本实现与官方公式使用同一几何约定，角度浮点值和阈值附近的舍入仍需用真实原件复核。[官方相对姿态计算](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py#L59-L177)

```text
python -B tools/vggt_co3d_pose_receipts.py --plan PLAN.json --plan-sha256 PLAN_SHA --truth TRUTH.json --truth-sha256 TRUTH_SHA --poses POSES.json --poses-sha256 POSES_SHA --output ANGLES
```

将 `ANGLES/pair-predictions.json` 及返回的 SHA-256 交给汇总器。该清单同时绑定真值外参、三组原始预测清单和转换器源码。模型推理、原标注转换与训练集隔离需由原件证据核定。此入口仅处理已获授权的现地外参，无需读取或转移图片、模型权重或历史训练输入文件。

本机通过 8 项 CO3D 回归，包含从三组外参到三个指标的完整链路。另用独立 4×4 逆矩阵、旋转 Frobenius 距离及平移叉积，对照 512 组随机刚体变换。最大绝对误差分别为旋转 `7.71e-12` 度与平移 `1.47e-12` 度。测试为合成 CPU 输入，实际检查点和留出结果仍待执行。

## 在实际设备上导出模型外参

`tools/vggt_co3d_export_poses.py` 使用已固定的 41 个原 VGGT Python 模块、原底座文件和本次评估对应的 `camera_head.pt`。用原生多帧裁剪预处理和 camera-head 前向，BF16 执行前向，FP32 解码外参。按实际设备核对 A100 CUDA 或海光 gfx936 HIP。海光两组先评底座，再在同一次模型驻留内加载训练头并评估；A100 从自己的训练头出发。保存实际 PyTorch、CUDA 或 HIP 版本及设备型号。[官方多帧预处理](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/vggt/utils/load_fn.py#L10-L81)

计划新增 `media_manifest_SHA256`。对应媒体 JSON 包含 `dataset_SHA256` 及相同顺序的 `sequences`，其中 `frames` 每项含 `frame_id`、媒体根下的相对 `image` 路径和原图 `SHA256`。传给预处理器的流直接来自已核验字节。原图或帧序失配立即停止，模型输出无效外参时按帧保存失败并由下游扩展到所有相关图像对。真值和三组外参收据须重复相同的 `media_manifest_SHA256`。

```text
python -B tools/vggt_co3d_export_poses.py --platform Hygon --plan PLAN.json --plan-sha256 PLAN_SHA --media MEDIA.json --media-root CO3D_ROOT --source-root VGGT_ROOT --source-manifest UPSTREAM_FILES.json --base-weights MODEL.pt --head CAMERA_HEAD.pt --head-sha256 HEAD_SHA --output NEW_OUTPUT
```

另一侧使用 `--platform A100` 和 A100 的实际训练头。两个目录各自的 `arm-files.json` 给出外参文件路径、校验值和实际训练头文件的身份。经许可在同一个证据位置使用三组现有文件时，按相对于合并清单的位置填写 `arms` 的路径，连同同计划绑定的真值交给逐对转换器。输出中记录训练头文件 SHA-256，正式验收需同时绑定产生该训练头的完整检查点及运行证据。现有目录和资产保留在各自获准的位置。

本机通过 12 项 CO3D 回归，新增验证帧序、媒体改动、路径逃逸、两组共享一个模型实例及收据兼容。部署文件按 Python 3.10 语法解析通过。推理部分使用测试替身验证交接，当前尚未在 A100 或海光运行此导出器，也未读取原 CO3D 媒体。

## 在原数据位置提前冻结留出集

`tools/vggt_co3d_freeze_holdout.py` 接收已批准位置上的官方评估格式 `*_test.jgz`、实际训练输入帧清单和测试图像根目录。库存 JSON 含 `coordinates=pytorch3d-world-to-view`、`training_frames_SHA256`，以及按类别排序的 `test_annotations`（每项有 `category`、相对 `path` 和 `SHA256`）。训练帧 JSON 含 `case=T12.camera-head`、可追溯的 `training_input_evidence_SHA256` 和 `frames`，每行记录 `category`、`sequence`、`image` 与 `SHA256`。该清单须覆盖参与三组比较的实际训练原件，冻结器仅读其路径与摘要。

默认按官方评估的随机种子 0、序列至少 50 帧和无放回抽 10 帧执行。`frame_id` 是该序列在官方注释数组中的索引。先排除实际训练序列并保留排除计数，再检查选中图像的真实字节 SHA-256。不同路径下与训练图完全相同的内容会导致冻结失败。原图缺失也会停止，不换选其他测试图。外参按官方 PyTorch3D 到 OpenCV 的转换写入真值。自定义 `sampling` 或排除训练序列后的类别范围按本次计划报告。[官方注释预处理](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/preprocess_co3d.py) · [官方取帧与坐标转换](https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py)

```text
python -B tools/vggt_co3d_freeze_holdout.py --inventory INVENTORY.json --inventory-sha256 INVENTORY_SHA --training-frames TRAINING_FRAMES.json --annotations-root ANNOTATIONS_ROOT --media-root CO3D_ROOT --framework-commit FRAMEWORK_COMMIT --output FROZEN
```

`freeze.json` 固定 `plan.json`、`media.json`、`truth.json` 及其 SHA-256，并记录计划图像对分母和排除原因。三份输入可直接接入上面的导出、外参转换和计分。路径和字节重合检查基于实际训练帧清单的覆盖范围，图像重编码或场景近重复需要独立证据。冻结器在数据现地生成证据，无数据下载或转移操作。

本机 16 项 CO3D 回归通过，新增覆盖相同输入的逐字节确定性、训练序列排除、异名重复字节、以及冻结到 AUC 的完整链路。测试为合成输入，尚未冻结实际 CO3D 媒体或评估真实检查点。
