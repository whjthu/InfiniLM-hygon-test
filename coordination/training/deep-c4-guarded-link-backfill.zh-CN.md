# I03 Deep C4 探针链接验证补位

父作业 `1103606` 的 `f13r2n01` 有一项冻结的一次性链接验证。它复用原 `1103606.156` 的 guarded probe.o 和原 `1103606.487` 的候选库，只验证依赖路径、13 个接口和被动 ELF。原预算为 4 CPU、8 GiB、0 卡、300 秒，两项上游编译已有独立终态。本次验证计划不含模型前向、权重读取或训练更新。

该验证已登记到统一 backlog。日常快照和派发交接使用默认工单，自动读取原意图。根需要核对这项验证的具体启动条件时，在本机读取原 SQLite 的只读视图。生成器复核 8 项清单字节、两个已封存的上游终态、节点归属、已有原意图及当前队列保留。

```powershell
$py = 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -B -X utf8 tools/prepare_training80_deep_c4_link_backfill.py
& $py -B -X utf8 tools/prepare_training80_dispatch_probe_args.py
& $py -B -X utf8 tools/prepare_training80_dispatch_handoff.py <original-receipt>
```

已有原意图或节点保留时，先对账。仅在 `WAITING_AUTHENTICATED_CPU_ADMISSION` 且获既有登录连接后，根可继续核对当次发布和实时放置，再经原 `main(publication, transport)` 调用一次。目标接收器再次检查父作业、编号步骤、两项上游账和唯一命名空间。启动器先写冻结原意图，之后 GPU 规划器读取默认工单并保留该节点。终态由原 CPU 被动链接独审消费后，更新该行并用新鲜设备证据恢复 GPU 准入。

该条目作为零卡验证登记，训练储备和可派 GPU 计数均为零。真实目标重新链接、运行时提供方、原七个权重范围及后续单卡验证需各自的证据。
