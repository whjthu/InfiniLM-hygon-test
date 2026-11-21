# VGGT 八对训练原请求对账

原请求 `c55668b8ef104a9cb18be34d94251351` 绑定父作业 `1087656`、节点 `f11r4n19` 和原发布 `1c70a9618539ae085fbd0c792abd71d1f7f069ed`。原本机提交记录为 `local-only/training-vggt-tum-eight-pairs25-recovery-v1-dispatch/request.json`，SHA-256 为 `32ed148dc6782010f6aa87e8faf29b1fec101c0b838ba863b21083f9bbabfe1c`。原回执的 SSH 退出码为 255。当时没有输出、原目录及 Slurm 记账行，状态保留为 `UNKNOWN`。

在确认已有认证的目标连接后，从目标项目运行一次只读 `python -B tools/reconcile_vggt_eight_pairs_original.py`。工具只读取该原父作业在原日期窗口的精确 `pre`、`gpu`、`post` 记账行和该目录中的小型回执。原意图须匹配请求、发布、放置和单次 1800 秒期限。独立 HOST 清理须在另外一个零卡步骤完成，并以原 `pre.log`、`post.log`、前后 PCI 清单、boot ID、KFD 和同 UID 设备句柄扫描交叉核对。

只有实际 GPU 终态、原请求绑定和独立清理都成立时，输出 `hold_retirement_evidence=true`。即使清理通过，原 GPU 失败或 collector 失败仍按原样保存。根协调者消费证据后才能处理该原保留，其他任务准入继续依照新鲜节点检查。缺失、歧义、`RUNNING` 或清理不足均保留原请求。工具不重复检查其他 VGGT 用例，也不会写回任务队列、提交步骤或读取训练媒体与模型数据。
