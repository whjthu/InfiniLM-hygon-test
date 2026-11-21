# Kimi r461 输出残差评分编译入口

原请求为 `e5ce43bacc53bc64c75a74005e5e249c`，父作业 `1103606` 的 O01 静态落点为 `f12r2n11`，申请 4 CPU、8192 MiB、0 GRES，整个窗口 300 秒。提交前检查两个父作业的实际归属、节点编号步骤、剩余时限和原标签唯一性。落点以该次真实检查为准。

从固定的 repo 包调用原渲染器，封存 13 个生成文件及其清单。容器只读挂载该目录与 CPU1113 producer，沿用已有 Deep gfx936 单次 controller 使用的 DTK 镜像和 setup。三个设备可见性掩码均为 `-1`，Slurm 明确请求 `--gres=none`。HOST 持久化 intent 和唯一 exec-attempt，再提交单个 Slurm 步。容器按同一启动 epoch 减少剩余预算，并为强制结束和对账保留时间。

实际目标调用前由协调者发布精确源码和认领。在目标 checkout 内运行：

```text
/usr/local/python3.12/bin/python3 -I -S -B -u host_once.py --claim-sha256 <CLAIM.json-SHA256>
/usr/local/python3.12/bin/python3 -I -S -B -u host_once.py --claim-sha256 <CLAIM.json-SHA256> --reconcile
```

第二条只读查询原步骤。尚无终态时保留 UNKNOWN，先前终态拒绝被后续不同的记账替换。实际 C0、同原 step 与 manifest、两份 fresh M 先于编译、每臂 M/MD 文件及依赖闭包、对象 ELF 头和原 `nm` 输出核验通过才返回 `COMPILE_VERIFIED`。独立对账的完整 stdout 可以直接作为原 `link_kimi_r461_output_residual_wave.py` 的 `--score-accounting` 文件，另传 stdout 内记录的 `score_receipt_SHA256` 和保留的 `<result-controller>/generated`。链接派发和资源释放仍要各自检查。

本机用正本渲染器生成文件，使用本机夹具提供 Slurm 和对象记录。验证包括原步骤 UNKNOWN 后同句柄对账、重复拒绝、过期 shell 在进入原 worker 前退出，以及编译命令或符号漂移拒绝。目标 Slurm、编译、设备数值和性能均待实测。
