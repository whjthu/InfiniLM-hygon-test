# VGGT 原训练派发前的设备准入复核

完成同一原请求的目标预检和本机队列覆盖后，取得新鲜设备回执。用同一回执和覆盖队列运行 `tools/prepare_training80_dispatch_handoff.py --select vggt-tum-bidirectional-native30`，取唯一的 `coordinator_handoffs` 元素。

由根协调者在现有 Python 派发入口调用 `tools/dispatch_training80_vggt_once.py` 的 `dispatch(snapshot, backlog, packet, transport)`。前三个实参分别为原已绑定回执、覆盖队列及刚取出的包。`transport` 为现有有界认证客户端的 `transport(body_bytes, stdout_file, stderr_file, timeout)`。该入口内部原子申领，交给它的包须尚未执行 `--claim-packet`。它保留原 VGGT 冻结启动器与 `1103606/f13r4n09` 的单卡资源范围，继续沿用原启动器的 1800 秒运行预算。

原启动器抵达本机传输回调时，保护层校验当次申领、原意图、原启动标记和传输正文，复核设备回执的本机单调时钟期限和启动锚点。证据齐全且设备准入已到期时，先独占持久写入原目录中的 `guard-pretransport.json`，再用其 SHA 标记为 `RETIRED_PRELAUNCH`。该原请求继续禁止重播。不同工作负载可用自己的新鲜设备回执申领节点。

在真正调用注入的传输前，保护层先将原预约持久登记为 `UNKNOWN`，再复核期限。回调前发生异常、证据有歧义或无法持久写入时，保留原预约。源启动器自己的 `attempt.json` 和 `transport.json` 继续保留原语义。远端执行状态仍须读取原 Slurm 记账及产物。实际终态与独立清理齐备后，沿用 `tools/prepare_training80_vggt_direct_finish.py` 生成原请求的 `finish` 负载。
