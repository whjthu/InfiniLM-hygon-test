# 认证恢复后的 80 节点准入检查

在本地读取冻结任务和启动器清单，汇总优先核对的原请求、节点和端口。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_reconnect_worklist.py
```

准备当次准入采集参数时，在本机读取最新训练队列、已冻结清单及原尝试标记。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_dispatch_probe_args.py
```

认证恢复后，根协调者从既有认证客户端调用 `collect_training80_dispatch_snapshot.py`，将输出里的 `collector_arguments` 数组逐项附给 `collect` 模式。采集器会优先检查训练候选所在节点，随后检查其他候选和常规节点。优先节点单独分批，各父作业的候选优先批次提交到同一个有界 CPU 探测池。持有原未决句柄的节点仍计入完整 80 节点 Slurm 账本，其设备记录保留为 `LOCAL_QUEUE_HELD`，等原请求对账清楚后再判定设备准入。多节点候选遇到任一持有节点时整项排除，对应的专用端口也不会加入采集参数。所有设备准入结论仍需实际 PCI、KFD、进程、端口检查和复查编号 step。

有待补位候选且需要尽快取回准入时，可给参数生成器传 `--priority-only`。采集器仍两次读取 80 台的 Slurm 状态，只对当次未出现编号 step 的优先节点发出 CPU 探测。回执 `device_probe_scope=PRIORITY_ONLY`，其余等待探测的设备标 `DEFERRED_BY_PRIORITY_ONLY`。若没有符合这两项条件的优先节点，当次回落为 `FULL_FLEET_FALLBACK`，覆盖所有可探测的普通节点。全池设备审计调用可省略此参数。

用 `receive` 按本机原始往返时间绑定收到的回执，立即交给 `prepare_training80_dispatch_handoff.py`。同一轮提交已取得准入且互不重叠的任务，未核验的其余节点随后补查。优先回执只能证明实际探到的设备准入。640 卡 busy 由另一次同步驱动采样计算。

以下是原候选节点。Qwen 原任务的端口 29918 在 Dev80 对账解除持有后才进入采集参数。

Qwen 的旧 Dev80 条目携带 `target_reconciliation_probe` 和依赖 SHA。认证恢复后从原登录端运行 `tools/reconcile_qwen_usgs_dev80_original.py`，将实际回执交给协调者。同一次查询两侧均须确认原目录没有出现、六个精确 Slurm 标签均无记录，且两台原节点均无编号步骤，才标记旧保留可解除。只要目录、在途记录或待确认清理出现，就保留原请求。旧 Dev80 始终保持禁止重播，待派的新训练仍需独立通过端口及 PCI/KFD/PID 准入。

| 节点 | 候选 | 首项操作 |
|---|---|---|
| `f13r1n10` | Qwen 原 88 条训练输入，两臂合计 30 次优化器更新，8 卡 | 查重叠的 Dev80 原请求、原命名空间和实际编号步，再验目标原输入和引擎。 |
| `f13r4n09` | VGGT 原两场景双配对双向训练，两臂合计 30 次优化器更新，1 卡 | 在目标核对双向四视图、模型、原请求唯一性和实际设备占用。 |
| `f13r1n19` | MiniCPM 原 88 条输入原生编码，8CPU、0 卡 | 查普通发布与独占范围，执行一次已采用的 CPU 前置验证。 |

VGGT 条目的 `target_read_only_preflight` 绑定登录节点上的只读入口和依赖 SHA。用原来源包在登录节点运行 `tools/preflight_vggt_tum_bidirectional30.py`，核对两次目标命名空间和 Slurm accounting、393 份训练框架源码、17 份原 TUM 媒体及模型文件元数据。程序不创建作业或目标目录。先完成资产检查，再采集新鲜 PCI/KFD/PID，避免把设备回执耗在文件核验上。若原请求标记或编号步出现，转入对账。模型文件本轮只核对已封存 SHA 所绑定的原元数据，实际运行仍核对设备和运行时。

`frozen_invocation_after_gates` 给出原启动器的参数结构。它需要已核验的真实发布提交、原任务准入和协调者现有连接。已有本机原意图的候选单列为待对账，并从下一步探测建议中移除。最新实际设备可用性和 busy 始终由新鲜的设备收据和驱动采样判定。预计卡数和 640 卡账本的配置容量仅用于排查。本工单只读取本地证据。
