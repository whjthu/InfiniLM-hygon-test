# BW1000 当前性能与后续方向

本页汇总2026-10-24北京时间已取得的结果；运行状态以带时间的独立观测为准。

| 模型/配置 | decode 总吞吐 | 资格及限制 |
|---|---:|---|
| Kimi TP8×PP4，B64/M16，32卡 | 133.59 tok/s | 先前已验收静态512输入/max128输出，约4.175/card；非单用户/在线速度 |
| Kimi TP8×PP4，B16/M4，32卡 | OFF61.305→ON61.857 | N4候选配对约+0.90%，完整token通过，但收益与波动接近 |
| GLM TP8×PP2，B16/M8，16卡 | OFF33.471→ON33.960 | resident K/V约+1.46%；未超过历史同形状34.003点值 |
| GLM TP8×PP2，B32/M16，16卡 | 历史43.50058 | 工作负载/轮次不同，不能与B16直接当代码加速 |
| DeepSeek tailSUM组合 | 不采用 | 第35索引最终token仍不同，纯B32控制与restore通过，QA18通过，ABBA跳过 |

Kimi B16最新first-ready用户均值3.126 tok/s，postanchor约3.877；二者计时窗口不同。在线动态服务应直接依据每请求token交付时间线，单独报告TTFT、平均decode与ITL尾延迟，不能把microbatch数当用户数。

已完成的Runtime诊断显示：预建views等可省约3–4us owner CPU，但多数M1/M4实际完成时间几乎不变；guard/cache-key/allocator微优化不是已证实的数倍加速路径。跨GPU的不同M观测有混杂，不能归因某个backend算法。

GLM真实layer38/真实权重、合成hidden测试约5.36–5.66ms热态wall、3.66–3.84ms owner CPU；gateup约1.09ms、down约.65ms对应低ownerCPU大项，attention prepare与row setup有明显CPU成本。stream事件含host发射空隙/依赖等待，不能把全部当纯GPU时间或PCIe延迟。单层结果不能乘层数作为模型极限。

截至2026-10-24 04:00北京时间，当前周期没有新增合格整模型吞吐纪录。GLM瞬态K/V组件已经完成单层及四层对照，配对中位加速分别约5.24%/5.50%；向量加载组件在M8约7.05%加速，M1 down反而变慢，因此只准备M4/8的受限整模型分派。逐专家M16矩阵化组件慢约5.52倍，已停止。组件倍率不能外推为整模型收益。

GLM整模型私有接入已经通过166增量CPU编译链接，单TU约9.745秒，23个未变对象按SHA复用。当前正在完成一次加载的B16/M8有限对照；尚未执行新整模型。Kimi实际Engine/broker测试582已FAILED1，32owner正常销毁、583四节点KFD空；动态阶段未到达，没有在线性能数值。失败静态门错误复用了不同batch几何的参考输出，且断言前未保存实际输出；下一次加载前先完成同几何、同调用轨迹的独立full-logit参考与token-only候选对照。

Deep685已COMPLETED0/18:31、8owner正常退出、独立audit0/postKFD空。在固定B32/M32 teacher36中，两个tile均开启的A全部token匹配；单独关闭projection或C4都使索引35从5078变为6533。两者都影响当前夹具，先保留；这不是普通自由生成验收或性能结果。没有新增15tok/s用户、在线goodput、投机加速或硬件物理上限结论。

## 词表头整模型终态补充

500已COMPLETED0/40:37，32owner关闭、独立501四节点KFD空。B1配对5.30629→5.34573 tok/s（约+0.74%）；B16配对59.83117→59.86904（约+0.063%），first-ready用户均值3.03590→3.03494。完整对照通过，但未建立明显整模型收益；组件3–5倍不能外推。B1只具同次加载matched reference，不能当独立历史gold；现有总吞吐纪录和15/user未改变。

在线控制原型已修复有界队列/事件与回收、8192 packed-token上限，并通过1000个CPU epoch；完整请求与censored prefix指标分开。实际跨节点broker/Engine已在582进入QA及静态校验，但未通过静态参考门，动态服务指标未测。

## 周期审核落实

Astra29已实际完成，审核451/163/159/456四轮，结论及逐项采纳/暂缓、负责人、资源和退出条件见`reviews/astra-review-r8381.json`及`reviews/astra-review-implementation-r8383.json`。685结果在审核水位之后收到，计为下轮第1轮；582重复已知不兼容参考门且无新增首差证据，记录为效率事故，不凑性能轮次。每3轮新结论审核；同路线2轮无收益、90分钟无可测进展提前触发。下一轮首先核对这些安排是否产生整模型结果。

## 04:25 北京时间执行补充

Kimi实际任务910375.588（32卡）及GLM实际任务910376.463（16卡）均已观测到RUNNING并进入权重加载。Kimi同一次加载做同轨迹参考/回放及真实动态到达测试；GLM同一次加载做瞬态K/V、向量加载、组合与基线对照。尚无新的性能结果。具体时间、句柄及限制见`evidence/bw1000/active-model-sessions-r8387.json`，后续跟踪同一任务，不因观察超时重提。

两项任务的实际日志均显示使用IBext_v8/SHCA高速网络，同时显示GPU Direct RDMA Disabled。这与回退管理网Socket是两回事；是否限制当前性能仍需profile和带宽/等待证据，不能凭初始化日志下结论。DeepSeek省去投影tile临时分配/拷贝的私有一TU编译687已通过，正在准备真实单层有限对照；尚未验证GPU收益。


## 05:06 北京时间执行补充（取代上文运行状态）

Astra30已完成并逐条落实，见reviews/astra-review-r8393.json与astra-review-implementation-r8395.json；每3轮全局有效实验审核，同路线2轮无收益或90分钟无可测进展、重复测试代码事故提前审核，独立实验继续。下一轮首先检查本轮卡片实际结果。

Kimi588已COMPLETED0，32owner正常关闭，590四节点KFD空，原审计命名空间错误保留、离线修复审计通过。有限8请求真实到达epoch的用户decode均值3.3552tok/s（2.5427–4.1341），TTFT均18.203s；含prefill服务窗有效post-first输出15.7158tok/s/32卡，绝非纯decode总吞吐。513job逐轨迹完整token/EOS/独立CPUargmax通过。最大ITL15.949s，不能只看pooledP99=.254s。现有日志表明早期最大gap与prefill区间重叠98.93–99.67%；下步仅测试欠满prefill最多等待10ms合到M4，固定外部到达时间与O,W,W,O，不盲重载。历史静态decode133.59纪录未改变。

GLM463因旧probe文件名在加载后失败，零性能arm，464清理空；修复468已实测RUNNING，继续同一次16卡加载的O/T/V/组合对照。ROWS16仅CPU源准备，按实际128MiB/layer、40层5GiB预算；是否进一步测试取决于当前profile，不以逻辑3.125GiB推断可容纳。Deep695单层copy-elision已COMPLETED0/2:36，8owner/audit0/postKFD空、24480stage哈希一致；逐step关键owner的ABBA点收益6.33%，低于OFF自身7.95%漂移，首decode贡献表观省时88.45%。保留默认OFF，暂不投入整模型加载。


## 05:32 北京时间执行补充

Kimi新合批对照594已实测RUNNING并进入四阶段权重加载，前置592完成0。GLM468首个完整128-token OFF锚为33.9883decode tok/s、26.2581含prefill tok/s、用户均2.02869，16owner完整输出通过；实际输入79–99token（共1412），**不是512输入**，当前任务未终态、候选尚未给收益。新ROWS16私有CPU库167已完成0/15秒，实际2TU13.33秒；尚无GPU或加载收益。

Deep679已有B32/M32、I512/maxO128两次普通生成全32×128历史输出匹配，18QA/8owner/audit0/清理空；整体mixed/B1候选仍拒绝。质量窗口离线decode观测57.28/57.91tok/s含CPUargmax/观察开销，不改原profile未请求、TPS置空的结论，也不把它当正式新纪录。下一必要加载将提供同几何正式计时与真实HIP关键路径，避免重复旧门。
