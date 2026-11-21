# InternVL USGS72 输入审计派发与回收

固定请求 `8110b01cd24846f8b6fcd08f0abda5a0`，父作业 `1087656`，节点 `a14r3n04`，8 CPU、12 GiB、0 GRES。源包已采纳 CPU 输入审计与一次独立终态消费。当前实际目标安装、提交及训练更新均为零。GPU、整模型及训练许可保持关闭。

认证、原命名空间和资源归属复查完成后，协调端调用 `tools/dispatch_internvl_usgs72_cpu_once.py` 的 `dispatch(publication, manifest_sha, transport_start)`。此入口使用已发布 source。原 launcher 完成固定数据和打包检查，创建原 `intent.json`。回调先刷盘该意图、在共享 SQLite 事务中核对同节点保留并记录 `DISPATCHING`，再调用已认证的原反向通道。回调不得提前送入命令正文。对超时、回执丢失或未见实际步骤号保留原请求及节点。

用已固定的独立 reader 消费真实的原 72 行结果和编号作业。只有实际终态以及 srun 和容器两个原控制器的清理链均闭合，`finish(reader_path, reader_sha)` 才以原请求结束 SQLite 保留。成功和已清理的失败均保存已完成、失败和未观察行数。原始私有状态仍留本地。目标端真实 144 次编码和 72 次 collator 的结果单独记录，这项 CPU 审计不计入训练队列储备。
