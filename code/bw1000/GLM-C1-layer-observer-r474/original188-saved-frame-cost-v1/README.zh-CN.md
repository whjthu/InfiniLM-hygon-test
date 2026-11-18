# 原188 saved48帧成本派生（待采纳）

此包仅首次派生已经保存的 marker 成本。root_GO/CPU_GO/GPU_GO/Model_GO/performance_GO 全为 false；未读取原report或owner，没有远端执行。root 与 Astra 审核、普通发布、翻明确CPU采纳位后，root 才能唯一调用。

准入是原已生成22337B report 的完整SHA，再由其31项 raw_artifact_SHA256 中取16个精确 owner 路径。每个 owner 最多128MiB，顺序执行一次稳定FD完整SHA及JSON解析，全部显式读取（失败也计）最多2GiB。历史原reader读取573299176B包含其他原件，不能当成这16个owner的确切体积；此包实际owner体积尚未知。物理IO受缓存和系统readahead影响，仍UNKNOWN。

复用原字节 attention_evidence.py 与 ffn_evidence.py，真实marker界157..440，78层和选中0/3层均核；只从 requests[1].observer_report 取一次，忽略重复保存副本。校验16个global rank 与0..7 local rank、3次实际AR阶段/past/sequence/capture token哈希。48帧代表3个连续且相互依赖的AR步骤在16rank的分片，3步和48帧均未建立统计独立性。实际步数仅在16份capture身份核实一致后记录；失败或不完整不预填实际3步。

host来自steady_clock秒，event来自CUDA event毫秒除1000。仅在同owner同frame同clock内作差；不跨host相减，不用host减event。输出各78层、FFN分段、attention occurrence/phase与indexer窗口；只算明确父子窗口交集以说明嵌套，不把相交窗口求和成独占GPU时间。微小event倒差沿原parser保持UNKNOWN，大负差/坏单位/缺marker拒绝。首末marker跨度不必覆盖完整AR调用。NoV曝光由原branch和tuple的源码关系推导，不是新增direct-call计数器。

资源为2个继承允许CPU（排0/1）、nice=max(inherited,10)、AS4GiB；AS不是硬RSS。无Slurm、新post、GPU/Native/Torch/SDK/provider39核查，禁止读取exact4失败/UNKNOWN接口。单180秒epoch在本机Git准入前确定，远端work留20秒，尾部同deadline硬退出；本机SSH在deadline前3秒停止并reap。SSH255或强杀保UNKNOWN，禁止重播。

未来唯一CLI（root发布后，source_review状态不能执行）：

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/GLM-C1-layer-observer-r474/original188-saved-frame-cost-v1/run_once.py' --publication <exact-published-40hex> --request-id <fresh-32hex>
```

R/results/GLM474-original188-saved-frame-cost-v1 是新排他namespace。cost-private.json 最多32MiB，原帧/输入/tokenIDs不出R；summary.json 最多32768B，同次stdout仅回该成本摘要及SHA。缺帧/校验失败仍留16/48完整分母与UNKNOWN，绝不重Model补采。此派生不是第二次整体验收reader，不改变原188质量、数值、性能或预约结论。

本机只新增一次合成时钟检查：正常、嵌套重叠、舍入UNKNOWN各1项；坏单位/负host/负event/缺marker/跨rank时钟5项拒绝。未重跑原18边界。见SOURCE_PROOF中的本地原结果SHA。

Astra增量修正只新增一次本机held-child清理边界：捕获可捕获中断、finally屏蔽二次信号，kill/reap自己的Popen，终止wait不调用可能抛错的left()。不重跑时钟或原reader边界。
