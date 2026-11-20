# 80 台资源补位交给冻结启动器

使用现有设备收据和只读待派发队列生成一次交接。各资源组分别显示 Slurm 无编号 step、设备已核验可补位、设备状态未核验、实测有持有进程、活跃 step 和本地队列仍预留的节点。`training_ready_deficit` 是距离 5 项未提交真实训练的缺口，开发集评分、推理和算子另记。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_dispatch_handoff.py <原收据路径>
```

根协调者准备提交特定候选时附加 `--select <candidate>`。该调用重新读取原收据、候选字节、原请求标记和 SQLite 预留。带提交意图、失去设备准入、申请重叠或收据过期时不会产生交接。交接保留两项父作业内的实际节点和卡数、原请求、发布提交、启动器和清单 SHA-256、现存队列预留 SHA-256，以及按原本机单调时钟算出的到期点。根协调者在同一原请求下执行现有独占派发流程，并确认直接派发的预留和目标端冲突检查。

`invocation.kind=coordinator_callback` 给出已经核过签名的 `main` 入参。由根协调者把现有审批通过的 `transport` 显式传入原启动器。`authenticated_host_python` 给出已冻结 HOST 启动器使用的三个标志参数，继续按原入口进行其目标端验证。带额外输入包、路径或手工绑定的入口出现在 `manual_launcher_binding_required`。闭合原输入后才计入 5 项储备，等待期间其节点仍可分配给有完整启动入口的候选。该工具只生成本地交接 JSON。
