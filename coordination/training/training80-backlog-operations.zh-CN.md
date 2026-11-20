# 80 台资源监测与训练待派发队列

资源监测智能体从当次认证连接取得两项父作业的编号 step 和设备准入收据。`tools/collect_training80_dispatch_snapshot.py` 提供逐节点的 Slurm、PCI、KFD、进程和端口观测。使用调用方原始单调时钟在接收时绑定收据。老收据可回看当时的编号 step，派发提议只使用有效期内的原收据。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/report_training80_backlog.py <原收据路径>
```

报告从 `local-only/tuning/queue.sqlite3` 只读提取在途预留，保留 `UNKNOWN` 原句柄的节点。`admission_by_zone_and_parent` 给出各组可提议、已保留或待查节点，`proposed_for_root` 给出互不重叠的候选。根协调者负责核对直接启动器的其他预留、再次确认实际归属并提交原冻结启动器。编号 step 的设备数和驱动 busy 分开报告，驱动 busy 使用另一次同步采样的原件。

数据和任务构建智能体维护 `training80-backlog-v1.json`。每项训练须有正数的实际优化器更新计划。既有数据的新独立回归也可登记，按原用例使用配方、盲测和质量阈值。开发集评分、推理、算子和下载独立列队。报告只把启动器调用已闭合、发布认领和任务前提齐备的未提交训练计入 `outstanding_training_ready`，并计算距离 5 项的缺口。缺少启动器入参绑定的任务列为 `manual_launcher_binding_required`。现场设备准入在派发时单独检查。任务一旦收到原提交意图转入 `reconcile` 或 `submitted`。

进入 `ready` 时在 `launch` 登记 `request`、`publication_commit`、`workload_SHA256`、`required_ports`、`authorization_key` 和原 `attempt_receipts` 的准确本机路径。用带 SHA-256 的路径绑定 `claim`、`manifest`、`launcher` 和三份收据。队列核对源码清单及其逐文件 SHA-256，包内启动器和认领也须列在清单中。三个收据由协调者在查验发布、目标前提和真实原命名空间后登记，收据的字段包含以下信息。

| 收据 | 记录字段 |
|---|---|
| `publication_receipt` | `request`、`state=PUBLISHED_MATCHED`、`publication_commit`、`claim_SHA256`、`manifest_SHA256`、`launcher_SHA256` |
| `prerequisite_receipt` | `request`、`state=TARGET_PREREQUISITES_VERIFIED`、`workload_SHA256`、`placement`、`required_ports`、`min_updates` |
| `original_request_receipt` | `request`、`state=REMOTE_UNSTARTED_VERIFIED`、`original_namespace_absent=true`、`accounting_label_matches=0`、`prior_invocation=false` |

原调用有本地意图、连接关闭或远端执行未定时，保留原请求并对账。空闲节点可优先补位真实训练，再从独立评分、推理和算子回归中选择已具备前提的任务。低优先级算子也需有可比较配置、有限时长和结果产出。
