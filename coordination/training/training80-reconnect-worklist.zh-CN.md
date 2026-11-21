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

有待补位候选且需要尽快取回准入时，可给参数生成器传 `--priority-only`。至少有一个启动条件已核、节点未被内部预约且请求使用加速卡的候选时，参数生成器才启用窄查。只有待预检任务、零卡工作或已被预约的候选时，参数保留优先节点和持有保护，并使用全池采集。采集器两次读取 80 台的 Slurm 状态。在窄查中，对未出现编号 step 的每个优先节点独立发出一个零卡的 CPU 探测，其他待探测设备标为 `DEFERRED_BY_PRIORITY_ONLY`。如果当次没有可探测的优先节点，回执使用 `FULL_FLEET_FALLBACK`。参数生成器的 `priority_only_enabled` 记录决策，实际 `device_probe_scope` 以采集回执为准。

用 `receive` 按本机原始往返时间绑定收到的回执，立即交给 `prepare_training80_dispatch_handoff.py`。同一轮提交已取得准入且互不重叠的任务，未核验的其余节点随后补查。优先回执只能证明实际探到的设备准入。640 卡 busy 由另一次同步驱动采样计算。

Qwen、VGGT 等训练分别完成目标预检时，可将各自生成的本机 `backlog.json` 汇入一份当轮队列。汇入程序从当前共享 backlog 核对范围，从各项收据核对同一份原始预检。冲突或证据失效的候选保留共享队列状态，其余候选继续进入同一次优先采集。已有提交意图的候选进入原请求对账，由该候选保留节点。输出路径须是新的 `local-only/` 文件。

```text
python -B tools/merge_training80_admissions.py local-only/<new-backlog>.json --overlay local-only/<qwen-admission>/backlog.json --overlay local-only/<vggt-admission>/backlog.json
```

将合并文件作为 `--backlog` 传给探测参数生成器、交接生成器和原子批次申领，统一使用其训练储备与优先节点。优先采集后，用一个新请求省略 `--priority-only` 补查其余节点，包含完全没有产生交接的情况。五项储备、Slurm 排队、真实运行和驱动 busy 分开记录。

以下是原候选节点。Qwen 原任务的端口 29918 在 Dev80 对账解除持有后才进入采集参数。

Qwen 的旧 Dev80 条目携带 `target_reconciliation_probe` 和依赖 SHA。认证恢复后从原登录端运行 `tools/reconcile_qwen_usgs_dev80_original.py`，将实际回执交给协调者。同一次查询两侧均须确认原目录没有出现、六个精确 Slurm 标签均无记录，且两台原节点均无编号步骤，才标记旧保留可解除。只要目录、在途记录或待确认清理出现，就保留原请求。旧 Dev80 始终保持禁止重播，待派的新训练仍需独立通过端口及 PCI/KFD/PID 准入。

将独立对账工具或下文 Qwen 训练预检通过原认证连接返回的成功 JSON 原样保存在本机，然后登记确认为未提交的旧请求。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/record_qwen_usgs_dev80_absence.py <authenticated-dev80-reconciliation.json>
```

完整证据登记在 `local-only/training80-reconciliation/qwen-usgs-dev80-original-absence.json`。此文件可以原样确认，内容不同则保留原件。探测参数、待派发队列和原子申领都会读取它。队列将旧 Dev80 显示为核验未提交，只有旧请求产生的内部保留解除。其他请求的占用、新鲜设备回执和实际端口检查仍在原流程中核对。缺失、无法读取或有冲突的证据继续保留两台节点。

Qwen 八卡训练还登记了 `target_read_only_preflight`。用原认证连接运行 `tools/preflight_qwen_usgs_exposure25.py`，它先查询 Dev80 原请求，再核候选训练的独立命名空间、CPU 输入步骤 1103606.490、393 份统一框架文件、88 行训练输入与 22 张训练图，并在返回前复查两个命名空间和 Slurm 归属。四片原模型权重和原初始参数在此只核文件类型与字节数，真正训练控制器仍核文件内容。取得预检结果后才按当前清理证据登记原 Dev80 的保留状态，再采集新鲜的设备、进程和端口 29918 证据。预检不提交作业，也不读取开发或盲测输入。

Qwen 领域训练原编号 GPU 步终态并通过 `read_terminal.py` 后，保存对应独立 Slurm 七列记账和原样终态 JSON。在原实验存储上用项目 Python 比较连续训练和恢复训练各自保存的第 25 步适配器。

```text
sacct -n -P -j 1103606 --format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES
python -B tools/read_qwen_usgs_resumed_adapter_values.py --accounting <实际记账绝对路径> --terminal-metadata <独立终态JSON绝对路径> --terminal-metadata-sha256 <同一JSON的SHA256>
```

两份适配器按各自原库存 SHA 读取，以张量名、形状、类型和全部值比较两份第 25 步存档。数值与文件序列化差异分别记录。优化器、调度和各 rank 的 RNG 恢复沿用同一原运行的独立进程证据，正式首步数值、性能及留出效果分别验收。

| 节点 | 候选 | 首项操作 |
|---|---|---|
| `f13r1n10` | Qwen 原 88 条训练输入，两臂合计 30 次优化器更新，8 卡 | 查重叠的 Dev80 原请求、原命名空间和实际编号步，再验目标原输入和引擎。 |
| `f13r4n09` | VGGT 原两场景双配对双向训练，两臂合计 30 次优化器更新，1 卡 | 在目标核对双向四视图、模型、原请求唯一性和实际设备占用。 |
| `f13r1n19` | MiniCPM 原 88 条输入原生编码，8CPU、0 卡 | 查普通发布与独占范围，执行一次已采用的 CPU 前置验证。 |
| `f13r1n12` | Gemma 原生编码及批整理，4 CPU、0 卡 | 核验目标训练数据与处理器元数据，用原框架对 72 条样本做一次 CPU 验证。 |

VGGT 条目的 `target_read_only_preflight` 绑定登录节点上的只读入口和依赖 SHA。用原来源包在登录节点运行 `tools/preflight_vggt_tum_bidirectional30.py`，核对两次目标命名空间和 Slurm accounting、393 份训练框架源码、17 份原 TUM 媒体及模型文件元数据。程序不创建作业或目标目录。先完成资产检查，再采集新鲜 PCI/KFD/PID，避免把设备回执耗在文件核验上。若原请求标记或编号步出现，转入对账。模型文件本轮只核对已封存 SHA 所绑定的原元数据，实际运行仍核对设备和运行时。

把原认证连接保存的成功预检 JSON 交给本机 `tools/prepare_training80_vggt_admission.py`。使用当前普通发布提交和一个尚不存在的 `local-only/` 子目录。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_vggt_admission.py <target-preflight.json> <publication-commit> <new-local-only-directory>
```

该入口核对已采纳训练包的 claim、原目标资产核验和两次原命名空间查询，并经普通 SSH Git 对照发布字节。成功时使用新目录下的 `backlog.json`，将同一路径以 `--backlog` 传给探测参数生成器和派发交接。采集新的设备回执，用 `--select vggt-tum-bidirectional-native30` 生成包，再走 [VGGT 带保护的单次派发](training80-vggt-guarded-dispatch.zh-CN.md)。该入口负责原子申领并调用冻结启动器。共享 backlog 保留为待准入。若认证、发布、原任务或设备任何一处有冲突，保留原请求处理对账。

VGGT 具备真实目标资产、任务授权和唯一原请求证据后，把任务纳入待派发队列。使用该次收据执行 `prepare_training80_dispatch_handoff.py --select <candidate>`，保存生成的单项 JSON，再用同一收据、`--claim-packet` 及清单中的原 `--intent` 原子认领。只有取得 `DISPATCHING` 且仍在原准入期限内，根才按 `invocation` 经已有认证连接调用一次冻结启动器。用当次实际原请求和可核实的步骤更新记录。连接断开或启动器返回时未能确认 Slurm 终态，保留原认领进入对账。

该 VGGT 原请求的独审入口列在 `target_completion_reader`。从父作业 `1103606` 保存原七列 Slurm accounting。独审在目标端读取 `results/training-vggt-tum-bidirectional30-v1`，使用认领包中的 manifest 与 claim SHA。终态结果与本次 accounting 使用同一份原始字节。

```text
sacct -n -P -j 1103606 --format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES
python -B tools/read_vggt_tum_bidirectional30_terminal.py <original-target-dir> <original-accounting> --manifest-sha256 <from-claim> --claim-sha256 <from-claim> --output <new-private-audit>
```

将原目标 `intent.json`、同一 accounting 文件和独审 JSON 保存在本地私有目录。用下方只读工具从 SQLite 读取原认领、核对原本机意图，再生成 `finish` 负载。独审退出码为 1 时仍可能有合格的清理证据，适配器保留原失败和全部 30 次更新分母。若 GPU 仍在途、原实际步骤冲突或缺少独立清理，适配器拒绝生成负载，保留节点。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_vggt_direct_finish.py <target-intent> <original-accounting> <private-audit>
```

把成功生成的 JSON 保存到私有新文件，由根执行 `tuning_queue.py finish <原认领返回的 run> <新文件>`。此步骤归还该原作业的节点并保留其请求及终态身份。源代码、测试和本机示例未产出实际派发或终态收据。

MiniCPM 条目的 `target_completion_reader` 保留在已有原尝试的对账视图。由 `tools/read_minicpm_native88_original.py` 获取实际 Slurm 记账，用原请求的 publication 和 epoch 绑定同一编号步骤，再读取原 88 条原生输入的结果。只在结果通过时，把它的文件与 SHA 交给现有目标训练资产准备入口。在途、未找到、失败和记账不一致都保留为独立状态。

MiniCPM GPU 原请求通过资产、去重和实时设备准入后，使用 `tools/control_minicpm_usgs_pair_target_v2.py --stage-sha <原资产收据 SHA> --source-sha <v2 源码清单 SHA> --publication <实际发布 SHA>`。源码清单见 [T05 v2 任务](../tasks/training-minicpm-usgs-exposure25-resume20-target-v2.json)。原请求、命名空间及三个 Slurm 标签一致，任何原尝试进入对账。v2 在实际 `srun` 参数中使用原 epoch 的剩余阶段预算并写入原阶段收据。原 worker、训练配方、25+5 次更新、独立终态和清场验收继续沿用。

SDXL 的 FP32 注意力反向首步对照使用 `tools/dispatch_sdxl_sdpa_adjoint_once.py` 的 `dispatch(publication, transport, snapshot=...)`。提供实际发布提交、既有有限 transport 和当前设备收据。入口核对 `1087656/f12r3n20` 的 PCI、KFD、进程和原编号探测步，在共享 SQLite 和原意图中绑定 1 卡后才调用一次 transport。调用结果不明时保留原请求和节点，由目标实际终态及独立清场解除保留。目标包先做零卡 PyTorch 验证，再运行一项完整首步诊断。该项单列验证队列。


Gemma 的 `target_read_only_preflight` 使用 `tools/preflight_gemma_render72_native_cpu.py --framework-root <verified_framework_root>`。入口先读取原数据包收据。若两处原命名空间均为空，先对账原 staging 请求，确认从未提交后再按冻结流程安排。若原 CPU 步、原意图或不完整的 staging 已出现，按原请求对账。完整数据包通过后，入口直接复用 CPU worker 的资产核验，核对 72 条训练输入、800 次曝光顺序、12 张训练图、14 份处理器文件和统一框架的审计模块。资产通过后再查 f13r1n12 的归属及编号步，复查 CPU 请求原命名空间和 accounting。

返回 `TARGET_ASSETS_VERIFIED_CPU_DISPATCH_PENDING` 后，用同一实际框架目录和发布 SHA 取得 `frozen_invocation_after_gates` 中的 CPU 控制器调用。控制器仍会校验现场归属和原提交唯一性。Gemma 与 MiniCPM 的 CPU 验证各自绑定本机原意图路径。提交尝试开始后保留节点，直到根协调者消费原终态并更新任务状态。CPU 前置验证和实际优化器训练分别计数。

Gemma 的 `target_completion_reader` 使用原派发所用的发布 SHA 作为 `--publication`。`tools/read_gemma_render72_native_cpu_original.py` 先查询 Slurm 原步，确认终态才检查原意图、控制器和 worker 收据，并按冻结数据的逐行哈希对照 72 条编码与整理记录。核验成功后，用报告中的 `worker_receipt.SHA256` 核对后续训练控制器所需的 `native-input-cpu-result.json` 原字节。CPU 前置不能代替八卡训练的原初始 LoRA 和实际 rank 采样器验证。

`frozen_invocation_after_gates` 给出原启动器的参数结构。它需要已核验的真实发布提交、原任务准入和协调者现有连接。已有本机原意图的候选单列为待对账，并从下一步探测建议中移除。最新实际设备可用性和 busy 始终由新鲜的设备收据和驱动采样判定。预计卡数和 640 卡账本的配置容量仅用于排查。本工单只读取本地证据。

Kimi retained28 的 0 卡编译使用 O01 的 `f12r2n10`。确认目标登录和当次归属后，由协调者调用 `tools/dispatch_kimi_r461_retained28_once.py` 的 `dispatch`，传入实际普通发布提交、项目内目标 checkout 和已有有限传输。该入口先在共享账本登记节点及唯一请求，再调用冻结 HOST 控制器。超时继续对账同一请求，原 wave 编译仍使用其独立节点和请求。编译和真实训练队列分开计数。
