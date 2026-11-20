# 认证恢复后的 80 节点准入检查

在本地读取冻结任务和启动器清单，汇总优先核对的原请求、节点和端口。

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 tools/prepare_training80_reconnect_worklist.py
```

认证恢复后，根协调者从既有认证客户端调用原 `collect_training80_dispatch_snapshot.py`，将回执中的 `collector_port_arguments` 附给 `collect` 模式。本批需要核对端口 29918，并关注父作业 `1103606` 中的三个节点。

| 节点 | 候选 | 首项操作 |
|---|---|---|
| `f13r1n10` | Qwen 原 88 条训练输入，两臂合计 30 次优化器更新，8 卡 | 查重叠的 Dev80 原请求、原命名空间和实际编号步，再验目标原输入和引擎。 |
| `f13r4n09` | VGGT 原两场景双配对双向训练，两臂合计 30 次优化器更新，1 卡 | 在目标核对双向四视图、模型、原请求唯一性和实际设备占用。 |
| `f13r1n19` | MiniCPM 原 88 条输入原生编码，8CPU、0 卡 | 查普通发布与独占范围，执行一次已采用的 CPU 前置验证。 |

`frozen_invocation_after_gates` 给出原启动器的参数结构。它需要已核验的真实发布提交、原任务准入和协调者现有连接。已有本机原意图的候选单列为待对账，并从下一步探测建议中移除。最新实际设备可用性和 busy 始终由新鲜的设备收据和驱动采样判定。预计卡数和 640 卡账本的配置容量仅用于排查。本工单只读取本地证据。
