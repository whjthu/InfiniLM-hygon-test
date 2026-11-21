# 80 台资源补位交给冻结启动器

使用现有设备收据和只读待派发队列生成一次交接。各资源组分别显示 Slurm 无编号 step、设备已核验可补位、设备状态未核验、实测有持有进程、活跃 step 和本地队列仍预留的节点。`training_ready_deficit` 是距离 5 项未提交真实训练的缺口，开发集评分、推理和算子另记。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_dispatch_handoff.py <原收据路径>
```

根协调者准备提交特定候选时附加 `--select <candidate>`。该调用重新读取原收据、候选字节、原请求标记和 SQLite 预留。带提交意图、失去设备准入、申请重叠或收据过期时不会产生交接。交接保留两项父作业内的实际节点和卡数、原请求、发布提交、启动器和清单 SHA-256、现存队列预留 SHA-256，以及按原本机单调时钟算出的到期点。根协调者用下方的事务入口在同一 SQLite 中取得派发权，再完成既有启动器的目标端检查。


把选中的 `coordinator_handoffs` 元素原样保存为本地 JSON，附加 `--select <candidate> --claim-packet <交接 JSON> --intent <冻结的本机原意图路径>`。申领在同一写事务中重新核对原回执、发布、任务范围和现有预留，并在提交前复查回执到期点。成功返回 `DISPATCHING`、原请求和可用期限。这个命令只保存派发权，原启动器负责实际提交。启动前继续核对期限和目标端归属。

若提交结果未知，保留原请求和节点，由现有队列的 `unknown` 或 `dispatch-state` 对账。取得原编号步骤终态和独立清理证据后，用 `finish` 提交同一请求、原意图、`placement_by_parent`、用例种类和证据 SHA。已终结的请求持续保留身份，节点可用于下一项工作。

直接启动的原任务即使没有 SQLite 行，也保留已登记的全部节点。带原意图标记的请求、已登记 `reconcile` 或 `submitted` 的请求，以及暂时无法核验原标记的请求会出现在 `original_launch_holds`。生成 `--select` 交接时重新读取这些本机标记，并绑定 `original_launch_holds_SHA256`。已登记的原提交证据缺失时，相关节点仍等待原作业对账，其他节点继续按现有证据安排。

`allocation_DCU` 表示该 Slurm 观测时点的父作业容量。`device_admitted_unheld_DCU_lower_bound` 表示新鲜探测中清除编号 step、PCI、KFD 和进程阻塞且没有本地保留的节点卡数下界。各候选的必要端口另行核验。`device_probe_scope` 标明当次设备探测覆盖范围，老收据或缺少设备收据时下界未知。`measured_busy_percent` 单独由作业内同步驱动采样测得。

`invocation.kind=coordinator_callback` 给出已经核过签名的 `main` 入参。由根协调者把现有审批通过的 `transport` 显式传入原启动器。`authenticated_host_python` 给出已冻结 HOST 启动器使用的三个标志参数，继续按原入口进行其目标端验证。带额外输入包、路径或手工绑定的入口出现在 `manual_launcher_binding_required`。闭合原输入后才计入 5 项储备，等待期间其节点仍可分配给有完整启动入口的候选。预览模式只生成本地交接 JSON。

候选源码、启动器或收据验证失败时，`invalid_launch_evidence` 给出该候选的原因。其他候选继续按正常优先级检查，可使用该候选未占用的节点。

SQLite 保存原子派发记录中已经绑定的 `workload_SHA256`。同一冻结工作负载换请求、候选或节点时，已有在途、未知或终态记录会出现在 `prior_workload_claims`，本轮排除重复派发。已登记的原节点保留按原记录处理。终态节点可以安排有独立证据的不同变体。原有独立核实的未提交证明继续沿相应原请求流程消费。

Deep inverse RoPE 的直接派发器取得本地预约后，若仍在 `DISPATCHING` 且传输前最后一次设备准入拒绝，先独占写入并持久化 `transport.json`，再将原记录标为 `RETIRED_PRELAUNCH` 并释放节点。保留请求、原意图和收据 SHA，禁止重播原请求。其他工作仍需自己的新鲜设备收据。进入传输前先持久登记 `UNKNOWN`，随后立即复查设备收据。登记之后即使传输超时、进程退出或收据过期，节点继续保留。没有成功写入本地未尝试收据时也保留原记录。

`training_pipeline` 区分本轮有合格设备可派发的训练、已通过任务准入但仍待设备的训练、待准备、待对账和证据错误。已提交和已终结计数来自队列记录。实际运行及等待资源的状态须查对应 Slurm 句柄。五项训练储备只计前两类，设备收据失效时本轮可派发数为零。
