# Graph64 按组补位扩展

原四节点复验 `1087656.687` 和 `1103606.684` 完成 32/32 卡、128/128 臂、512/512 驱动样本并确认独立清场。本扩展使用相同的 M184、K4096、N1024 BF16、原按 PCI 派生种子、两套驻留输入、同一权重、预分配输出和 64 次 GEMM 的 eager/Graph 对照。worker 相对原 32 卡包只有 placement 模块的 SHA 变化。每节点资源沿用 12 CPU、64 GiB、8 卡，计时内无 DMA 或集合通信。原结果见 [32 卡复验](../../../results/bw1000/2026-11-22/O01-Graph64-32card-original687-684-four-node-paired-measurement.json)。

由根将本轮已知的 protected 节点及 fresh CPU-only receipt 交给 `package.prepare(publication, campaign_epoch, snapshot, protected, requested=None)`，取得计划、有限 HOST 代码和正式发布检查路径。requested 为空表示从该收据中挑全部可选节点。每台必须是清晰的整机、原父同组、真实 8 PCI、KFD/PID 清洁、CPU 探测句柄已知且没有编号步骤。每片最多四台，同父作业同账本组。实际计划卡数由这次提交前的选择得到。根持有各片独立的请求、预约和在途账本；未得原终态与独立清场不得释放。目标传输前后应按原请求对账。包本身不写共享队列。

统一用原 epoch +60 秒开始，在 +0、12、24、36 秒运行四个 8 秒臂，worker 在原 epoch +125 秒截止，Slurm 每片 140 秒。成功启动的 client、实际 numbered Slurm 行、已通过的卡和采样分别保留。失去准入的片记录原分母与未启动原因，其他片仍可推进。跨节点无独立时钟误差界时，只报各节点真实局部窗口及 pooled 样本，不将共同排期当作集群 simultaneity。

使用目标原 staged `controller.py --reconcile` 获取各片实际账号、源从属、数值证据和完整分母。协作让出时，`controller.py --inspect-stop LABEL SHARD_REQUEST NUMBERED_STEP` 只读核原 intent、真实 srun PID/startticks/UID/argv、原父归属、当前精确 numbered 步及成员、实际 RUNNING 行和所有 worker 已安装 USR1 的原意图。通过时返回只匹配该编号步骤的信号命令。根只在真实训练准入冲突及发信前复查一致时使用。未知、改号、缺失或错误属主不生成信号。发 USR1 后仍保原失败数、实际终态、零卡清场和预约。
