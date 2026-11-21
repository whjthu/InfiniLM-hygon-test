# 节点内八卡 HIP 双向搬运扩展

先导 `1103606.634` 的原独立对账是准入输入。扩展固定排除这台先导节点，覆盖其余最多 79 台、632 卡。先导与扩展的证据可组合成 80 台覆盖记录。全机队的 GPU 计算 busy 仍须独立同钟采样。

扩展逐字节复用 [已通过先导的 v2 worker](../fleet-hip-copy-duplex-v2/SOURCE_FILES.json) 和原 DTK provider。每台仍为 8 卡、12 CPU、12 GiB。所有训练优先节点、活跃步骤、已保留或未知的原请求先退出。每片只落在一个原父作业及同一组，最多四台。独立片占用和归还用独立请求标识，同一扩展只允许一次有限发送。目标 Slurm 再检查本轮实际成员和编号作业。出现新占用时停止该片的派发并留下原决策。

根调用 `launch.ship(publication, fresh_snapshot, original_pilot_reconcile_bytes, existing_finite_transport)`。目标已封存的路径下执行 `controller.py --reconcile`，输出精确 Slurm 行、分片或缺失状态、每台有效卡数、原结果和逐臂日志 SHA、独立重算方向流时长和八卡墙钟。总体资格要覆盖分片的原分母。Slurm 无记录时保持原未知。每片完成或失败后，只有原 Slurm 终态和之后实际独立的零卡、同 PCI/KFD/PID 清场均可证实，才按现有队列接口归还。父作业、新请求与不同分片保持独立。

每片 Slurm 最多 150 秒。同一原 epoch 的 worker 截止为 180 秒，收集器为 225 秒。实际卡数取编号 Slurm 记账，合格卡数取原 owner journal。源文件与小结果可发布，内存 payload、原日志留实验端。
