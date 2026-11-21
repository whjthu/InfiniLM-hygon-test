# O01 GLM 编译原请求保护

绑定唯一请求 `b9a7f0b78c566911ed40004fc906ce44`，父作业 `1103606`，节点 `f12r2n13`。原 HOST 控制器使用 4 CPU、8192 MiB、0 GRES 和 300 秒。目标范围、命名空间、编译计划及原终态核验沿用冻结包。

该零卡 `validation` 行已经登记到统一 backlog。根先核验原反向隧道认证、HOST 普通发布以及目标源码所在的原项目目录。执行端导入 `tools/dispatch_glm474_fresh_kv_cpu_once.py` 并调用 `dispatch(publication, remote_repository, transport)`。`publication` 是已发布的 HOST 包提交 `b06cec29ee58f2837c59a95554cec65931c5b85b`。`transport(argv, stdout, stderr, timeout_seconds)` 接收原已认证通道的回调，将参数数组传给原目标端，输出流保存在专用本机私有文件内。

启动入口在共享 SQLite 写事务内核对 O01 现有保留及本机原标记。没有冲突时独占创建 `local-only/GLM474-fresh-kv-pack-gfx936-CPU-v1-original-dispatch/intent.json`，刷盘并提交 `DISPATCHING` 保留后才调用回调。GPU 补位和独立调优队列都会保留整台 `f12r2n13`。SQLite 冲突、缺少登记、包字节漂移或需要对账的原标记都会阻止进入回调。

回调退出码与原 Slurm 结果各自保留。响应丢失、超时、返回 0 或本机中断时继续保留原请求。恢复后核对原目标意图及精确 Slurm 标签。使用冻结 `host_once.py --claim-sha256 <原认领SHA> --reconcile` 核对原实际终态，并独立确认资源清理后，以原 `queue_run`、原意图、`placement_by_parent={"1103606":{"f12r2n13":0}}` 和 `validation` 结论走现有终态入口。原意图暂不可见时保留请求。已核实终态归还节点并保留原标记及私有输出。该条目不增加训练储备或 GPU 利用率。
