# Kimi 可选本地流式服务适配 r9118

CPU305 已通过八条 Deep1033 真实生成 token 的 HTTP/SSE 回放：全文和有用 token 流均 8/8 一致，另 14 个 Unicode/EOS 边界样例通过。这是传输验证，不是模型性能。

`incremental_text_r9118.py` 固定 cleanup=False，解码边界永不后退，缓冲未完整 Unicode，并在 EOS/输出上限刷新尾部。只有已提交的非 EOS、非 padding token 计入有用吞吐。

`resident_api_r9118.py` 提供 localhost OpenAI 兼容子集：单 user 消息、greedy、stream=true、有界 prompt/输出/请求队列。HTTP 线程只校验和排队；模型操作始终由主线程调用。断连后只能在合法事务边界结束当前推理。

当前冻结的 r9116 没有外部流式 sink，因此尚未接入真实 Kimi 模型。不能将读 case-end JSON 包装成 SSE 后宣称实时流式服务。后续应从现有 emitter2 已验收输出，经已有 collect/socket 到 leader0，再调用本模块 emit_tokens；禁止逐 token NFS。调用顺序和完整合同见 resident-api-contract-r9118.json。

此模块是可选项，不阻塞 state-only 首个真实结果。生产资格仍须真实客户端 TTFT、ITL、用户 decode、排队/拒绝/失败/未完成、自然 EOS 与输出质量。

六档客户端矩阵已准备：C=1/2/4/8/16/32，另有 Poisson32 请求。当前 B1 事务串行，客户端并发不等于 native batch。CPU308 用实际 HTTP 回归了六档，实际峰值分别为 1/2/4/8/14/20；CPU 回放很快，不能把请求设置32当作实测同时32个模型用户。所有 CPU 吞吐均不作为模型指标。

后续命令：`python client_load_r9118.py --trace arrival-trace-r9118.json --topology ACTUAL_TOPOLOGY.json --endpoints http://127.0.0.1:PORT/v1/chat/completions --output UNIQUE_RESULT.json`。必须将实际服务分配的全部 target/draft/aux 和每个 DP 副本写入拓扑，若 DFlash2 与 Modal 各占一张卡就计两张草稿卡。默认输入示例仅用于准备，未连接模型端点。

当前推荐 APIb：resident_api_r9118b.py 增加服务器排队耗时。新协议可用 committed_event_wire_r9118.py（默认关闭）发送固定 JSON token 事件，不能传任意 Python；发送阻塞耗时计入真实 decode。不要修改当前冻结 r9116。
