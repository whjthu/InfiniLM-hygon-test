# Astra review 128 — 2026-10-29

近期 **0 项新增可证明整模型/真实服务收益**。三轮诊断有价值，但不能将CPU、编译或profile当成加速。结论：GLM做一次同加载clean C1 ABBA；Kimi先单卡实际组合子集/生命周期；Deep先完整HOST ABI与实际C1组件。完整逐卡字段在同名JSON。

**最新用户资源优先级覆盖上述技术排序。** root已通过 `read_thread` 核实“智能体支持”最新人类指令：先修复Deep SSE、GLM实际权重P0重验、Kimi身份审计与三模型C2 HTTP/SSE隔离验收。G128A/K128A/D128A均保留源/CPU准备，GPU在接口验收后再安排；本审核不触发任何新性能GPU提交。原性能goal继续保留，不标paused、不丢弃当前证据。以下耗时为未来实际安排时的执行估计，不是现在启动承诺。

## 127落实与时间成本

|127卡|实际落实|判断|
|---|---|---|
|G127A|787跟同一实际句柄完成，532在CPU原端恢复审计，未重跑GPU；16owner各3个实际T1窗口|完成，下一步做干预，不重复observer|
|D127A|同入口数据合同修复后1195完成；4/4正确自然EOS、0cap、normal/postempty|完成，旧1187失败保留|
|K127A|533源码合同/CPU构建32.31s完成，Native06db|工程进展；实际生命周期待证|
|K127B|保留456 C1/C8，未为registry追加整模加载|继续采纳|

Deep1195总10m52，其中加载447.72s；剩余204.28s混合暖机、测量、质量和清理，不能全记实际GPU测量。GLM532恢复1s；789/790各64次CPU写总0.731/0.741s，不含排队启动。Kimi533只构建32.31s。准备/排队及GLM787加载分项未见完整证据，写UNKNOWN。下一轮结束分别记录准备、排队、加载、测量、质量、清理；不新增报告框架。

## G128A：支持一次GLM同加载clean C1 ABBA

这是最直接的整模型因果机会。787 clean C1用户p50 4.958/4.956 tok/s、ITL p50约201.5ms。stage0 rank0首attention约1.269ms、其它rank约14.665ms；rank0在forward前同步写progress。789/790在实际计算节点、相同目录，synthetic atomic write中位11.396/11.487ms。时序和量级足以支持一次最小干预，尚不能证明磁盘造成等待，更不能认定网络或attention GPUbusy。

复用r171b：routine至多每秒写一次，first/BEGIN/DONE/ERROR强制，原路径、payload、60s watchdog不变，同owner线程，默认OFF。**当前源仍保留原三arm observer_overhead与固定索引paired_quality；先补齐四arm plan和汇总，所有observer=false，再执行。** 这属于完成可运行候选的必要小修，不应重建框架或追加observer。限频不是后台心跳，长forward继续遵守原60s stall语义。

原GLM owner在既有隔离16卡TP8PP2边界一次加载，沿原暖机后A/B/B/A，每arm相同4QA/预算，16执行仅4unique。记录每请求(n−1)/(last−first)、TTFT、ITL P95/P99、task/16卡与Native wall，并留写次数/skip/force计数。各arm正确答案/自然EOS/0cap与正常退出；不加exacttoken门。

两B改善需超过本次A端漂移和请求离散，四arm完成即判。无收益、仅约1%或混合结果默认OFF并park，不另load追微小差异。明显改善只能称该C1 pilot收益，未证明在线服务或广泛质量。单次11.4ms若处于202ms关键路径约5.6%仅为假设尺度；不能乘16owner，也不能直接相加两stage。CPU/runtime补齐估10–20min；Model首判按历史加载25–35min再加约3.5–6min四arm，排队未知。

## K128A：先实际单卡subset probe，fullfit继续UNKNOWN_STOP

采纳已有 `Kimi-r170-next-device-card-r171.md`。原Kimi owner完成同MoE TU生产helper/probe与同一真实registry的一次HOST构建后，在已授权单卡边界做一次有限验证。真实同时持有92组packed bank、93层cache、92字段共享1workspace，逻辑子集46,291,641,856B≈43.112GiB；不读未初始化FP4，不weights/forward/comm，不以一次分配乘92。

沿原卡检查24MLA+69KDA零初始化、同owner共享与跨owner隔离、正常和host异常lease退出后的event完成/pattern、重入拒绝、弱引用释放与新generation。实际free保持4GiB floor及原预留；中途不足保留实际完成数量安全退出，不能缩小样本称通过。一次全子集、正常/异常各一次与原负例足够，不造暖机性能样本。准备估15–30min，GPU首判2–5min，排队另列；原inner180s/Slurm4min边界保持。

通过只更新持久子集实证账。完整模型参数、load/finish staging、临时算子空间、CCL及真实forward仍未覆盖。1189/1191实际省126MiB可复用；Runtime销毁后146MiB残差归属UNKNOWN，不称完全回收或leak。新增每I96同步的性能仍UNKNOWN。本卡不承诺Model fit，也不测吞吐。共享寿命失败则停共享、保留cache exact独立实现与456 TP16PP2基线。

## D128A：正确生命周期和当前C1路径先于整模加载

1195 OFF CAPI中位320.75–326.59ms；observer增加5.0–12.9%。每owner实际43次reference attention，SWA/HCA/CSA=2/20/21，batchedFP32 attention调用为0。attention包络174–201ms，内含cache/compression/index/metadata约100–120ms，routed约20ms。包络含提交与等待，不能当GPUbusy或把100–120ms全部归给weight Cast。

保留r172 C128两不可变BF16权重在load后转F32的最小候选，原F32 GEMM、hidden/output边界和smallT OFF不变。直接成员cache、Runtime owner、原checkpoint和成功generation确定身份；两转换/drain成功后发布，before-ready/重复load拒绝，sequence reset保留cache，异常先drain，析构正常释放。策略仅在quiescent边界切换。A/B都驻留cache，故defaultOFF仍新增每owner320MiB；实际allocator/峰值另测。完整重编所有complete-type HOST消费者，沿现有9TU/对象绑定清单，禁止单TU混旧ABI。

CPU闭合后原Deep owner只做一次8owner matched C1组件：tokens=requests=1，production rows128、真实512×4096权重，四实际pasts385/395/437/407及既定ratio128边界邻居；不拿历史r8697 rows32/past575..639或prefill/B32替代。每case O/N预热、2个ABBA周期，状态重置在timer外，包围CAPI含原必要copy/sync/drain，以8owner慢者及各owner实测判断。finite/既定容限、score/pooled/state/cache/commit/reset合格；全20HCA每decode40投影，单层probe按实际层数计数。

weight Cast可影响wall份额仍UNKNOWN。组件质量/寿命失败或无稳定收益即park，不付完整Model加载。只有C1组件有足够价值、memory可接受才进入下一次observerOFF Model配对。CPU准备估15–30min、完整HOST编译实际未知（沿900s截止）；组件5–10min加排队；后续Model历史加载约7.5min，当前不支付。

smallT1112已8/8→7/8，维持STOP/defaultOFF，不能再次包装纯开关为新候选。Kimi registry/state_dict专用reload、GLM重复observer同样停止。root逐卡登记源/CPU采纳、GPU因用户接口验收优先而暂缓、owner及退出条件；接口验收后再分配性能GPU。独立源码准备可继续，不为本报告等待。
