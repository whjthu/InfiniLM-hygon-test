# MiniCPM 检查点 20 的 SDPA 重复性对照

从原检查点 20 各启动两个默认 SDPA 和 math-only 进程组。顺序为 A1、B1、B2、A2，每组八个 rank，每个 rank 只做一次原生零更新前向。四组共用冻结的输入、权重及各 rank 恢复 RNG。组内以 profiler 的实际 SDPA 调用确认分流。结果保留 32 次前向的分母和缺失证据。完整训练恢复、吞吐和留出效果另行验收。

用 `prepare.py <新私有目录>` 从公开提交还原旧控制器和依赖，再生成本轮启动器。输出 `source_SHA256` 是生成目录中 `SOURCE_FILES.json` 的校验值。用 `check_local.py <同一私有目录>` 核对生成的入口和失败路径。

在目标端将新私有目录的相对路径结构原样保留。运行其中 `host_once.py --asset-preflight --source-sha256 <生成清单的 SHA-256>` 可以在保底作业仍运行时核对原配方、检查点库存和新请求名称。该模式只读，不创建 Slurm 步骤。找到重复名称时用同目录的 `host_once.py --status` 对账。

真正派发时使用本次发布后的采用记录和新鲜的设备准入。原保底交接完成后，`--preflight` 检查当前 T05 所在的父分配、空节点及新命名空间。`--preflight` 与 `--launch-once` 均要求 `--source-sha256`、`--admission`、`--admission-sha256`。启动器自行取得八卡 PCI/KFD、端口及独立收尾证据。连接断开后用原请求对账。
