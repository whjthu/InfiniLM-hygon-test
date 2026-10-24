# BW1000 当前性能与后续方向

## 第46次审核已落实：2026-10-24 18:05 北京时间

769、510、772三轮均实际结束并各消费一次。Kimi激活并行分配输出完全一致但64个配对全负，rank0完整provider3.375→5.182ms，STOP；无spill/动态索引waterfall证据，不能把原因猜成已证。Deep C128缓存720组有效值误差为零，但max-owner方向混合、端点漂移10.7–37.9%，收益UNRESOLVED，默认OFF，不因表面1.4倍开整模型。

GLM510完成完整基线128+诊断128、QA、16normal、审计及511清理。16卡TP8PP2/B16M8/I512O128基线decode39.0436tok/s、prefill113.893秒、含prefill输出12.3420tok/s；emitter用户均1.9213tok/s。这是重复基线而非代码提升。已命中16个事件窗；两层都是dense层，不代表routedFFN。rank0层0/1同层8row attention stream累计416.025/412.872ms，ownerCPU410.753/407.339ms；FFN在TP前293.791/293.525ms。诊断prefill149.285秒明显受观察/控制扰动，不能视为纯算子占比。

原owner已并行收到三张实现卡：GLM真实producer完整prefix provenance删除重复CPU元数据；Kimi单一融合反量化+有界BF16 W13 scratch，先核真实F32输出能力且含cold成本计时；Deep只一次原生chrono计时澄清。新GPU包仍需具体源/资源明确后执行，未把CPU准备当速度提升；最终token/EOS验收保持，目标仍未完成。以下为历史路径。


## 第45次 Astra 审核已落实：2026-10-24 17:00 北京时间

本轮763/639/766各消费一次，下一周期计数归零。审核明确：新增整模型已验收优化收益仍为零，不能用编译、工具通过、局部逐位恢复或组件倍率代替。当前Kimi插桩基线decode136.6367 tok/s/32卡；真实emitter首末token用户均1.3816、最高约2.1361 tok/s，最大ITL约76.58秒，prefill期间的等待仍明显。完整profile见 `../../evidence/bw1000/Kimi639-profile-compact-r8677.json`。

两项已经交原owner并行实施：Kimi实际W13 <2,4>归约后八次串行activation改为wave0八lane并行，保留原累加树/route/W2，先单TU CPU构建与有限provider配对；GLM原两TU补显式prefill事件窗，cohort0/1分别选stage0层0/1，保留原stream/drain/所有数学。各自20分钟CPU首判目标，具体GPU包尚待准备，不把held320DCU当利用率。审核不再阻挡这些实现，不增加SDK/freshness或全网络中间bit门；整模型晋级仍检查完整最终token/EOS与实际计时。

766真实整层0组合已正常闭合、8rank目标输出与重复稳定；串行4次512行到合批2048行的完整适配调用137.232→118.632ms，成本下降13.55%，含reset/传输/调用摊薄，仅为组件信号。当前绑定仍component-only，且不同于91层MoE MXFP4 Down，暂缓独立整模型加载。Deep760负路线及GLM盲M4扩展继续停止。以下为带时间的历史记录。


## 当前插桩基线已闭合：2026-10-24 16:40 北京时间

Kimi639 已 COMPLETED0/27:23；32卡正常退出，所有完整输出IDs与前后QA通过，原post清理通过。TP8×PP4、B64/M16、每请求输入512/输出128：两次普通decode总吞吐136.7207/136.5526，均值136.6367 tok/s（4.2699 tok/s/卡）；prefill均85.6827秒；含prefill的输出吞吐均56.4304 tok/s。普通阶段仍加载hipprof interposer，因此这是当前插桩基线，不能据此宣称相对历史版本优化提速。四节点日志IBext_v8=true、Socket_fallback=false。已取得闭合trace，原owner正在CPU分解实际关键路径。

Astra44后763与639共2轮；8658真实整层组合及有限无capture成本测试已获执行授权，源认领73f10b2，不把准备当完成。GLM8692工具已实际8子进程trace可见、507正常结束/508清理通过，仅工程进展；当前B16/M8 prefill profile包CPU准备中，尚无新Model性能。以下段落保留时间标记，作为历史状态。


## 第44次审核落实：2026-10-24 16:25 北京时间

Astra44已完成，757/760/501各计一次并消费，四项决策见 `reviews/astra-review-implementation-44-current.json`。后续763是下一周期第1轮：8卡61秒、8normal/原audit/764节点清理为空；原Row与合法拆分路径均复现完整O/N，preTP目标全部相同，仅按512行分段TP归约即恢复8rank目标。这与757 GateUp512因果点分别成立；整层组合/最终token/EOS/吞吐尚未验证。最小8657组合CPU准备在进行，保留原classlayout/Common/SDK。

Deep760同load B64M32/B32M32都完整128IDs/EOS/QA通过，CAPI52.4351/54.1665tok/s，B64容量比低3.20%，driver E2E低0.94%，按无收益规则结束，不反序/不采纳。单首pair不证明所有负载普遍无益，旧699约58.18纪录不覆盖为54.17。GLM501 O8合格39.1894tok/s；N4全16行最终分歧，仅6行首token即分歧，其余11–48，拒绝N4性能。16normal/原audit0/502清理通过；CPU复用原首token logits定位，token编号差距不能代表数值误差大小。

Kimi639当前183/servingpolicy1 B64M16实际RUNNING（08:19:42UTC加载11:26），独立短hiptrace将分析实际kernel/CCL/PP/host区间。普通测量仍有hipprof interposer，属于instrumented baseline，不能冒称历史unwrapped提速。GLM8690工具探针504已终态但child FileTooLarge导致无有效trace，root505补清理为空；先CPU修工具限制，未授权GLM新整模型profile加载。本周期没有新增整模型提速或15tok/s用户验收。以下为历史路径。


## 第44次审核落实：2026-10-24 16:25 北京时间

Astra44已完成，757/760/501各计一次并消费，四项决策见 `reviews/astra-review-implementation-44-current.json`。后续763是下一周期第1轮：8卡61秒、8normal/原audit/764节点清理为空；原Row与合法拆分路径均复现完整O/N，preTP目标全部相同，仅按512行分段TP归约即恢复8rank目标。这与757 GateUp512因果点分别成立；整层组合/最终token/EOS/吞吐尚未验证。最小8657组合CPU准备在进行，保留原classlayout/Common/SDK。

Deep760同load B64M32/B32M32都完整128IDs/EOS/QA通过，CAPI52.4351/54.1665tok/s，B64容量比低3.20%，driver E2E低0.94%，按无收益规则结束，不反序/不采纳。单首pair不证明所有负载普遍无益，旧699约58.18纪录不覆盖为54.17。GLM501 O8合格39.1894tok/s；N4全16行最终分歧，仅6行首token即分歧，其余11–48，拒绝N4性能。16normal/原audit0/502清理通过；CPU复用原首token logits定位，token编号差距不能代表数值误差大小。

Kimi639当前183/servingpolicy1 B64M16实际RUNNING（08:19:42UTC加载11:26），独立短hiptrace将分析实际kernel/CCL/PP/host区间。普通测量仍有hipprof interposer，属于instrumented baseline，不能冒称历史unwrapped提速。GLM8690工具探针504已终态但child FileTooLarge导致无有效trace，root505补清理为空；先CPU修工具限制，未授权GLM新整模型profile加载。本周期没有新增整模型提速或15tok/s用户验收。以下为历史路径。


## 第43次审核与执行：2026-10-24 15:17 北京时间

Astra43已完成，三轮新结论只计一次，计数归零；逐卡落实见 `reviews/astra-review-implementation-43-current.json`。Kimi8648四组真实单层对照与条件GateUp候选继续；GLM仅prefill4/decode8、Deep新B64M32/B32M32两条独立配置路线并行准备具体源码，尚未启动它们的新GPU作业。Kimi高batch实际profile/AUX只搭载下一次必要加载；B128、EP新框架及重跑DeepM64继续停止。

GLM498已完整验收：两个16卡TP8PP2实例，合计32卡，真实共同collector窗口decode **76.53036 tok/s**、每卡 **2.391574**，E2E **24.45962 tok/s**。32个独特I512/maxO128请求，完整IDs/EOS/QA、32owner正常退出、原audit0、499四节点清理全空；IBext_v8无Socket数据回退。逐用户decode平均1.90614、最高2.43360tok/s，TTFT均值94.6373秒，未达到15/user。与旧PP4 B128的70.15负载/镜像不同，不称纯并行策略因果提速。约39秒初始ITL停顿和慢prefill仍待解决；prefill4可能缩短TTFT/全prefill而加长早用户初始gap，必须测真实时间线。

Deep754 B64M64第31个token失败，无性能晋级；B64M32未执行，原因仍不能隔离到M64或B64几何。新B64M32是独立有限35分钟配置试验方向，不重播754；首配对收益≤3%停止。Kimi751离线有效定位仍为denseMLP输出，8646CPU35秒构建不是TPS收益；旧Kimi133.59和Deep约58.18纪录保持原范围。以下内容为带日期历史路径，旧RUNNING状态不代表当前仍在运行。


## 第42次审核落实：2026-10-24 14:34 北京时间

Astra42已完成并逐条落实，报告与执行见`reviews/astra-review-42-current.json`及`reviews/astra-review-implementation-42-current.json`。本次提前审核消费495/636两轮；随后751离线恢复形成新的有区分力定位，计为下轮第1轮。498/754原作业继续，不加新的审批或无关门禁。报告修复仅CPU，不为补JSON重载模型。下一次必要且兼容的Kimi高batch加载才合并AUX/actualprofile验证；EP不是当前实现的配置开关，源码可行性已明确STOP新native/GPU。

751实际8卡组件运行FAILED1/90秒，原因是多线程共享报告在序列化期间改变字典；8worker正常关闭、752节点清理为空。预建6目录中实际只有Oa/Na/Nb/Nca/Ncb五次捕获，Ob未运行，O重复未知。离线逐点核验五次8rank数据完整：Oa复现621观测，两个N对各自重复一致；183修正Nc恢复全部KDA及FFN输入边界，首个剩余差异是layer0 **denseMLP输出**，随后exit。真实denseMLP顺序为GateUp → SituAndMul(beta=4,linear_beta=25) → RowParallel（含TP归约），仍不能区分这几段；不能称MoE原因或整模型token修复；没有新TPS。

GLM498（32卡、两个独立16卡实例）及Deep754（8卡、B64/M32vsM64）已实际RUNNING/加载，只有带时间的最后观测，尚无新性能结果。Kimi636已公开：prefill72.75→203.58inputtok/s、含prefill8.478→12.096outputtok/s、puredecode15.758→15.743不增；尾ready用户均值1.47667→1.79641、TTFT31.694→13.588秒。旧B64M16的133.59decode纪录保持独立。下文为历史路径，不能把历史RUNNING当当前状态。


## 最新闭环：2026-10-24 13:46 北京时间

GLM495（16卡、TP8/PP2、B16/M8、真实I512/maxO128）两轮O/N/N/O完整通过。decode均值37.9131→38.7834 tok/s（+2.2955%），每卡2.36957→2.42396；prefill72.4527→72.4729 inputtok/s基本不变，含prefill12.2866→12.3784（+0.747%），逐用户decode均值1.87580→1.91177。完整token/EOS、QA、短输入续写、16owner退出、原审计0和496节点清理均通过；网络实际IBext_v8，无Socket数据回退。仅保留已测范围的opt-in小收益/defaultOFF；不称全batch极限，约39秒初始token间隔与长prefill仍未解决。详见`../../results/bw1000/2026-10-24/glm-shared-validation-qualified.json`。

Kimi636是固定M512的既有PP延迟反馈调度资格测试，最后观测仍在加载，尚无新增性能。原633在CDLL前因历史receipt路径错误失败、634清理为空；605b222仅机械修复并保留失败证据。同时准备真实Embedding+原layer0的小组件CPU测试，未授权GPU。GLM下一步CPU评估两个独立两机实例与四机PP4的每卡吞吐，尚未实测，不能把独立数值相加称集群吞吐。

Astra41已完成并落实，495计为其后第1轮。全局每3轮有区分力实验审核，连续2轮同路线无收益、90分钟无可测候选/因果证据、严重回退或重复可避免工程事故提前审核；编译与轮询不凑轮次。下文是历史路径，运行状态以本段及带时间的status为准。


本页汇总2026-10-24北京时间已取得的结果；运行状态以带时间的独立观测为准。

| 模型/配置 | decode 总吞吐 | 资格及限制 |
|---|---:|---|
| Kimi TP8×PP4，B64/M16，32卡 | 133.59 tok/s | 先前已验收静态512输入/max128输出，约4.175/card；非单用户/在线速度 |
| Kimi TP8×PP4，B16/M4，32卡 | OFF61.305→ON61.857 | N4候选配对约+0.90%，完整token通过，但收益与波动接近 |
| GLM TP8×PP2，B16/M8，16卡 | I512 OFF31.635→ON37.990 | 最新472完整验收，resident约+20.09%；prefill基本不变 |
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


## 06:17 北京时间：审核落实与新实验启动

Astra31已完成并逐卡派发，参见reviews/astra31-decisions.zh-CN.md；新增699/702/594三个有效实测轮次后，Astra32已实际启动。审核与已准备好的独立测试并行。

GLM468现已完整验收：短输入79–99token、B16/M8、16卡的配对decode从33.4505提升到35.3069tok/s（+5.55%），用户均值1.9990到2.1050；不是I512或全batch纪录。真实I512的rows16/驻留单变量包5722已逐SHA上传，06:17唯一controller实际启动；同一次加载先短输入回归再长输入O/R/R/O，watcher先就绪，无新性能结果。

Kimi594已COMPLETED0/36:27，32owner正常关闭、1666job同轨迹oracle通过、595四节点KFD空。O1/O2完整输出一致；W1/W2有3请求不同，O对W1有7请求、O对W2有4请求不同。W不采纳。服务窗提升不能覆盖token失败；预声明all-decode区间反而从O约29.16/29.24下降到W28.24/28.38。下一步CPU首个原生分歧定位，不重载只重现差异。

Deep699已COMPLETED0/20:57，8卡B32/M32、I512/maxO128，正式prefill108.5963/108.0601inputtok/s、CAPIdecode57.8535/58.1816outputtok/s，driverdecode56.8257/57.1425；质量、审计及清理通过。它是同几何正式基线，不是新增代码收益；单用户CAPI约1.81tok/s，采样步有6.494倍开销，不能拿采样空隙当普通利用率。

整数702已在8卡完成96个精确CPUoracle/guard测试并独立复核：寄存器MMAC的单卡达成INT8中位736.05TOPS、INT4中位1440.03TOPS。仅是整数指令算术参考，非物理峰值、完整GEMM、FP4或模型加速。


## 07:10 北京时间：GLM 实测收益与并行推进

GLM472已COMPLETED0/49:28，16owner正常退出，唯一独立audit0及postKFD空。真实16个I512输入、maxO128、B16/M8，配对O/R/R/O的decode均值31.6355→37.9903tok/s（+20.0876%），每卡1.9772→2.3744；含prefill11.5798→12.3373（+6.54%），prefill72.7472→72.8213inputtok/s没有实质改善，用户时间线均值1.6092→1.8801。I512完整最终token/EOS匹配同次OFF，不能冒充独立原模型数学证明或I512/O512。短79–99输入配对34.5297→38.1137（+10.38%）另列。详见results/bw1000/2026-10-24/glm-rows16-resident-qualified.json。

B32/M8的真实32条I512夹具162、私有6TU库168已CPU完成；下一轮先配对吞吐，减少全128step仪器采样等待。Kimi实际599为32卡首decode原生边界因果定位，尚无新性能结果。InfiniLoader典型16.99GB原始分片已单片预置并完整校验，首次copy含源SHA45.79s、目标verify17.45s须单独计费；不把它当整模型加载加速。Astra32已落实，新增707及472共2轮有效实验，达到3轮或提前触发条件后再审核。
