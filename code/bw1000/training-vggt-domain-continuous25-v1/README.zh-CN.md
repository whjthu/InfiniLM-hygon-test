# VGGT 双训练场景 25 步与新进程恢复

在统一框架 `a68952cb2285d6593b665147249d14ec1f8cecbc` 的原生 VGGT 后端上接入两个训练 RGB-D 对。原后端 SHA256 为 `49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c`。原模型、在线 aggregator、camera_head、BF16 损失、AdamW 和裁剪等计算链全部复用，数据收集器改为两个固定场景，输出增加领域数据说明。

输入顺序固定为 `freiburg3_structure_texture_near` 和 `freiburg3_nostructure_texture_far`，各一个双帧对。每步消费一个双帧 batch，零起游标为 `(global_optimizer_step - 1) % 2`。本次 25 次更新使用 near 13 次、planar 12 次。在线处理两场景，各自有注册深度和有效掩码。第二个场景的深度来自固定 Voxel51 转制，已通过几何检查，尚缺该场景原 TUM PNG 的逐字节对照。原 near 场景有一张深度原 PNG 对照，另一张来自同固定转制。数据适用 TUM CC BY 4.0，保留 IROS 2012 作者引用和转换来源。[TUM 许可](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [转制说明](https://huggingface.co/datasets/Voxel51/TUM-RGBD#parsing-decisions)

`data_contract.py` 接受两个精确 manifest、各自数据根和三个开发 manifest。构建前核验 14 个已绑定文件及两个 manifest，输出 17 文件私有 bundle，拒绝媒体相同、路径逃逸、未冻结来源和开发组入训练。开发 cabinet/teddy 的清单只参与媒体 SHA 去重。household 保持来源级隔离，没有读取盲测媒体。原始图像、位姿与私有 bundle 留在 `local-only` 或目标数据域，Git 维护源码。

原配方的 `warmup_steps=5` 为计时预热，随后 `max_steps=20` 为计时更新，共 25 次真实 optimizer update。学习率为固定 `2e-5`，没有独立可变调度器。实际常数学习率在原 optimizer param_groups 内保存，恢复时由原完整 optimizer state 复原。梯度累积为 1，每个检查点都在完整更新之后，未完成的梯度没有跨检查点积累。

检查点在全局步 10、20、25 保存。原后端使用 `vggt-full-optimizer-step-v2`，包含 camera_head、完整 AdamW state、optimizer tensor placements、Python/NumPy/Torch CPU/device RNG、输入源与完整 batch 张量身份、步数及 `next_batch_cursor`。先验证完整 integrity hash、状态形状及优化器绑定，再应用并检查恢复后的整边界。恢复过程异常会 poison worker，并要求退出。新 overlay 没有修改这些函数。

`native_worker.py` 在同一个计算步骤内每次只运行一臂，根协调者依次用三个独立 Python 进程运行 `continuous`、`resume10`、`resume20`。各进程重新加载原模型，相同数据根和 backend 进入原生训练函数。两个恢复臂读取 continuous 的已写 receipt 及原检查点 SHA，分别真实恢复更新 11–25 与 21–25。合计 45 次执行更新，其中正式连续训练 25 次，恢复对照重复 20 次。原内部 in-process recovery 关闭。外部对照通过 PID/startticks 验证进程不同，并比较完整步分母、数据游标、loss、模型/优化器/placement/RNG 哈希及最终状态。loss 绝对容差预先固定为 `1e-6`，状态哈希要求相同，失败原样保留。目标为单 rank、单 HIP 卡。

根协调者应先按 393 文件清单核验独立的完整统一框架副本，再将 `domain_overlay.stage` 生成的四份 delta 放入 `generation/src/gencuda`。其中替换一个 backend 并新增三个辅助模块，保留其他 392 个原文件，共 396 件。每臂命令接受 `--arm`、`--data`、`--weights`、`--upstream`、`--output-root`、`--bundle-sha256` 和 `--overlay-sha256`。共享 `VGGT_DOMAIN_EPOCH`、`VGGT_DOMAIN_DEADLINE`，支持最大两小时的单个原始时间预算，子进程不会重置它。output-root 下各臂目录必须全新，数据绝对路径三臂相同，输出目录不会进入原 checkpoint resume identity。提交器负责前后设备占用核对和 Slurm 终态，并在目标机校验实际 Torch/HIP provider 及冻结源清单。

本地 `check_contract_CPU.py` 使用两个真实数据对跑现有几何门禁，验证 canonical 更新和检查点源码字节、实际 metadata/cursor 函数，以及比较器的合成边界。CPU 测试没有运行 Torch 模型、完整 checkpoint state 或设备。实际 CPU 原件记录了 21 项坏契约被拒，初次比较器 fixture 共享 Python 字典导致损坏同时影响两臂，已将 fixture 各臂隔离并重核。所有设备训练、真实恢复、可用模型产物与开发效果仍待目标端运行，正式 CO3D 和三轮正式性能资格继续各自验收。
