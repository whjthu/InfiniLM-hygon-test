# Astra review 119：两轮内取得当前真实负载的性能证据

结论：本轮没有新增 qualified 大模型性能收益。工程资格工作应收敛为最小修复；Kimi 的性能归因不能继续全部等待 2K MMLU 容量恢复。条件采纳 root 新提供的已通过 1K Native305726、原 GSM32、C8W8/flatT64 路径，单独观察当前真实 verify；2K Native e955 的容量问题独立处理。GLM 继续 910376.767 原句柄，消费稀疏 profile 后只做一个最大 critical wall 区间的候选。Deep 最新1134已终结失败，转最短数值/状态分离诊断，不列吞吐收益。

本审核仅本地读文件和写本报告，未 Git、SSH、集群调用、上传、编译或 GPU。权威快照为 closure-state-root-r9360-a.json；RUNNING 仅代表该快照观测。末尾 1K 资源/源码分支来自 root 审核期间增量指令，尚未由本审核验证实际新 candidate 构建或执行。审核不阻塞在途工作，也不授权中断、热改或重放未知状态作业。

## 118 的落实与最近实际结果

| 前审卡 | 实际落实 | 119 决定 |
|---|---|---|
| K118A 显存失败定位 | 478 长 prefill 4 GiB reserve floor 少约 60 MiB；12/18 case 留存。pool 两 HOST 已编译；482 在首个 forward/trim 前因亲和性 fixture 检查失败 | ADOPT r134b 最小生命周期修复；trim 效果与首次峰值修复仍 UNKNOWN。不得把482登记为trim负结果 |
| G118A 当前全模型稀疏归因 | r131 observer/r133b entry 已落地，767 TP8/PP2 16卡 RUNNING，controls-before/profile/controls-after 一次加载 | ADOPT 消费同一句柄；不新增同类 profiler，不重做 M15 微矩阵 |
| K118B verify 优先 | 历史与478诊断证明 verify CAPI 约88–90% cycle；当前实输入细分尚无结果 | 改为条件 ADOPT 已qual 1K C8W8 当前路径，避免2K容量串行堵塞；不拿历史层fixture充数 |
| D118A Deep 原前缀资格 | 1131 暴露 owner 线程未保持 CUDA current；r135 scoped restore，CPU462 1 production HOST+4 mock 通过；最新1134 FAILED1，首次数值断言0.0453034、casecoverage空、8owner正常close/postempty | ADOPT 15分钟定位实际tensor/比较语义；不放宽阈值冒充缓存正确，不直接重跑16case；无 MTP head/自然文本质量/tok/s 声明 |
| TP32 full93 禁止加载 | CPU457 Common依赖漏链，460 self-contained方向失败；464复用对象显式链接Common2e14，约2秒/0新TU成功 | ADOPT 正确公共provider链接；constructor仅oneKDA/oneMLA/head/all93cache，不是full93权重/forward。full93继续UNKNOWN_STOP |

GLM764 最后可用配对结果仍为 task 13.203829→13.326042 tok/s（+0.926%），decode 22.723909→22.914066（+0.837%），user mean 1.164790→1.177654；16卡 task/card 约0.8252→0.8329、decode/card约1.4202→1.4321。M15组件约4.5倍不能替代这些 whole-model 数字。两epoch/arm、32题小筛查不足以证明普遍稳定收益或正式质量不劣，M15保持 default OFF。

Kimi已测33卡口径：C1 total decode17.2465、/card0.5226、user p50 18.0354；C8W8 38.7052、/card1.1729、user p50 5.3457；C16W4 45.4744、/card1.3780、user p50 2.9426。C1为32个顺序请求，不是32并发；p50>15不等于所有用户>15。这些是既有emitter结果，不是HTTP服务。Deep暂无合格吞吐结果，Qwen0.6B公共工具HTTP闭环不能计作三大模型性能进展。

## 工程耗时的结论与收敛办法

本次新增失败均未形成性能轮次：482 fixture前置检查、TP32链接、Deep device-current修复。主要风险是有卡、有报告，但缺少能测当前实输入的源码。已知CPU462约15.86秒、464约2秒；482 GPU步骤80秒/keeper104秒失败；GLM764历史26m57s，478历史target41m46s。准备/排队/加载各自精确耗时缺失，写UNKNOWN，不能把全部耗时称GPU计算，也不能拿累计报告数当迭代数。

立即复用同父版本 actual object、MD headers、provider和既有成功基线。每个owner只交一个可执行最小候选及一个对照；15分钟内给出source可行性结论，未能绑定就STOP该新probe，回收方向而非增加一层清单。新增计时只需沿现有结果记录准备、排队、加载、测量、质量五段；不新造审计框架。90分钟无新增因果证据触发路线复核，不自动取消在途作业。

Kimi482的源码结论已修正：actual_submit.hpp SHA 8c0bd70a... 与CPU458 MD一致；RuntimeWorkspaceHandle构造时调用apply_once，Runtime::create或独立CCL barrier并不保证已完成CPU绑定。旧fixture既假设rank*4物理编号，又在首个Ops前检查；482未记录实际mask，不能断言发生了物理映射变化。r134旧前置width4源码PARK。r134b在forward前只检查ownerID/非空allowed subset，原MoE+原sync之后、计时窗口外检查width4/disjoint/stable，记录实际mask；不手工绑核，不改production native/数学/trim，不增加GPU操作或sync。本地8个mirror检查仅是源码资格，远端实际1HOST构建尚待闭合。

## GLM 稀疏结果的决策树

先使用现成r133的资格：两个吞吐指标controls drift和profile effect均<=3%，各epoch32 EOS/0cap/0unfinished，小筛查不退；16owner自然T16×3、T15×3、B8 PREFIX/B1 PREFIX覆盖，缺测为UNKNOWN。单owner原stream events最多512；读取发生在原有drain后，不增加sync。不要把源码本来存在的decoder sync误称observer新增sync。

1. **先判是否可解释。** 匹配同sequence及完整(row,past,count)、stage/rank；prefill与decode分开。单owner相邻events可给有界区间，host与GPU不能相加，PP两stage不能相加；跨主机绝对host时钟没有同步证据时不得相减。profile超扰动或无自然selector时只保留诊断，停止本轮收益判断，不加padding补样本。
2. **算子路径。** 如果同stage多个rank的attention或FFN都慢、等待边界较小，先选占实际critical wall约10%以上的最大区间。当前detail只有每stage一个full/shared层的RMS、attention、FFN；它没有直接分离W13/W2、路由、专家直方图或内部CCL。不能把单层时长乘层数。attention最大就对当前实际full/shared路径做一个kernel/布局候选；FFN最大先复用实际router/job形状和现有FP4实现，最多补一个代表性调用的W13/W2/collective边界，随后只改最大phase。
3. **专家负载路径。** 只有现成或极小附加观测拿到同调用的有效token/expert计数、实际kernel/job划分、各rank用时后，才区分工作量失衡与相同工作量执行慢。统计max/median、空expert比例与有效计算量，不设一个未经实测的通用阈值。路由计数不齐就UNKNOWN；不能为了拿直方图新增GPU→CPU同步。若输入/路由复制于TP各rank，不能凭rank时间差认定专家分配失衡。只在真实job不均衡解释尾rank时试一个工作分组候选，不更改routing/topk以制造均衡。
4. **CCL路径。** 有collective专属begin/end并能核同sequence发起进度时，才区分到达偏斜与传输本身。等待长且此前compute偏斜大，先修上游尾rank；所有rank准备接近、collective仍长，再查实际payload/拓扑和可合并调用。当前FFN粗区间包含什么尚须源码/细分证明，不能用FFN总耗时认定带宽不足。无可行动的专属界限就停通信优化猜测。
5. **PP路径。** stage1 incoming长必须与stage0 production/发送顺序和stage自身空闲区间对应；这可能是上游计算或调度依赖，不是可直接删去的通信。若stage失衡产生可重复bubble，且当前调度源码具备安全边界，只试一种平衡/提交调整；不删依赖、不重分配完整模型权重来碰运气。若两stage都计算占主导，PP重排无已证收益。

下一候选只针对一个经上述分流确认的大头；若最大可改区间不到约10% critical wall、无法证明可移除、或观测本身失格，STOP该路线。GLM user约1.17距15非常远，小于1%的M15局部收益不会解决服务目标；所需实际吞吐曲线与质量仍必须测量。

## 下一轮实验卡（每模型至多两张）

**K119A — 2K容量最短判定，ADOPT；原Kimi pool owner，独立于性能主线。** 假设：修正fixture后，可测native pool在原长prefill中的first-live peak与释放后retained，并确定trim是否足以恢复reserve。复用r134b、CPU458对象、原实际MoE输入；仅1HOST fixture改动，CPU准备预计<=10分钟，排队UNKNOWN，组件加载UNKNOWN，GPU测量<=10分钟。原379两节点16卡组件资源，在root批准且与其他作业隔离后进行2warm+原4 ABBA，不重跑已完成的18case。先检查forward基本正确/有限、生命周期、原floor/清理，再看每owner first-live lowwater和after-free retained，timed span不含新CPU mask检查。首峰仍越floor、释放不足或归因不明则2K Model PARK；after-helper trim释放不能证明首峰fit。只有确定修复且最坏原case有余量才用原失败case资格，再续未完成24题，保留不同run provenance；不删题、不降4GiB floor。whole-model性能收益UNKNOWN。替补直接转K119B，不再等待此卡。

**K119B — 当前C8W8 target verify归因→单候选，条件ADOPT，Kimi性能第一优先。** 假设：当前flatT64的target内有一个主导且可改phase，而非draft/guard开销。按root增量分配，复用已qual 1K Native305726和原完整GSM32，独立380四节点32target卡+371 n03一draft卡；与379组件隔离，不改2K e955或在途进程。owner先用15分钟核actual objects/deps、真实1K输入完整性、当前W2pair路径、可无新增sync插入events的位置；无法快速提供当前Model instrument就STOP，不转历史单层fixture。原GSM问题和模板不得截断以适配1K，若上下文资格不覆盖实际题长则此分支不执行。

第一轮：同一次权重加载，固定预先选定的少量完整真实C8 cohorts，OFF/ON/OFF；保留自然EOS/cap和全请求结果，不为采样截短回答。2warm后至少4个自然flatT64 verify窗口，分别记stage的MLA/KDA、routedMoE/sharedMLP、TP/PP和host-submit边界；仅现成输入、原stream、有界event，原drain后读取，不开逐段sync PROFILE_R9125，不加D2H路由统计。观察开关仅用于诊断扰动：两个OFF漂移与ON相对OFF均<=3%，质量基本合理/任务答案与parse不明显退，缺测记UNKNOWN；测量目标<=10分钟，加载参考历史约28.5分钟但不是新配置承诺，排队UNKNOWN。此轮合格就能说明当前C8哪里慢，即使没有加速也有因果证据。

第二轮仅在第一轮给出可改大头后执行：复用现有FP4 W13/W2pair或实际attention/collective源码，只修改一个主导phase；四对真实cohort配对，报告效应、配对漂移和区间。若candidate早已具备私有DSO可在受支持边界选择，使用同加载；否则下一次最多一次加载，不能为复用权重热换DSO。收益必须超过本次配对漂移并无基本质量/容量/状态退化，才扩大GSM32及正式质量benchmark；小样本仅筛选。收益未超漂移或大头不可改则STOP，不盲grid。总wall、prefill、decode、TTFT/ITL、逐用户>15达标数/goodput、/33card均报告；token可不同，按实际有效token与完成时间计算。C2/W8只在同加载低成本且已验证路径可接入时补充近SLA点，不能替代C8/T64归因。

**G119A — 消费767，ADOPT；GLM原owner/当前16卡。** 单次controls/profile/controls原计划保持；排队加载已在途，不重新支付。结果到达后本地分析目标15分钟，按上述树确定一个候选或明确STOP。输出只需实际最大区间、覆盖、扰动、质量和可移除量，不造新profile。退出条件沿r133冻结策略，未完成/缺测不外推。

**G119B — 一个最大critical区间候选，条件ADOPT。** 仅当G119A合格并定位约10%以上可改区间，原owner用当前actual math对象做一个patch。复用相同32arrival、TP8/PP2、一次加载内ABBA四个完整episode（两对，筛选而非正式统计结论）或等价预先固定成对设计；准备上限30分钟先给可测候选，加载/排队UNKNOWN，测量预算<=30分钟；不得为满足时间预算截短episode。预热和原基本质量检查保留；跨token差异按实际吞吐和配对任务质量评分。全模型收益须超本轮control漂移并正常退出才扩展benchmark，否则保留default OFF且STOP。替补是消费Kimi真实性能结果，不再跑M15八offset。

**D119A — 1134终结后的最短分离诊断，ADOPT；Deep原owner，先零新增GPU。** 最新service-and-Deep-root-r9362-a.json覆盖旧在途状态：1134 FAILED1、约7m13s，8owner normal_close/postempty，AssertionError(0.04530340983426637)，casecoverage空。该数值是GPU_run_r135.py:85–86的continuation logits relative RMS比较形式；实际traceback/张量与首次case仍由owner核实，不能仅据空数组认定任何case都未执行。源码:92在构造append参数时先调用compare，断言发生便整条case未追加，已完成的baseline/unpublished/committed检查可能因此没有留下结果。device-scope修复已跨过先前阻塞，不等于43层事务全部通过。

15分钟内读取现有实际traceback、case位置/accepted counts、actual/reference shape、dtype、normalization定义、双方调用序列与reset/committed lengths；确认比较的是相同后续输入和相同已提交前缀，分开数值差异与事务/owner/length/causality/cancel/reset硬安全。0.01是当前harness诊断门槛，不是用户正式质量标准；不直接上调阈值，也不把4.53% RMS差异自动等同答案错误或可接受。最小源码修复是先保存case-start与已通过的状态证据，再计算并保存完整数值诊断/failed标记，最后决策；顶层异常保留当前case部分记录。下一次资格如获授权，先仅复现该原case；只有硬安全全部有效且差异已证明是独立数值诊断时，才在显式reset成功后收集其他独立原case并保留overall数值失败，避免首个诊断抛异常吞掉后续16case证据。任何状态/因果/非有限或reset异常仍立即fail-stop，不继续未知状态。

复现预算<=5分钟测量，加载/排队UNKNOWN；若无法解释比较语义或state差异，STOP该路线并保留缓存错误嫌疑，不花下一轮完整矩阵。原4GiB floor、owner和清理保持，无GPU F64。原16case完整覆盖仍是未完成事项；不能以正常close替代通过。即使数值诊断收口也需自然QA/真实benchmark才能晋级模型质量；没有MTP head，仍无spec性能声明。该卡不挤占Kimi性能probe。

## Spec、量化、资源与理论边界

C8W8与C16W4初始flatT64相同，full-active verify均约0.936–0.940秒，cycle约1.045–1.061秒；后者每人有用token/cycle从约5.895降至3.587，拆给更多用户解释了总吞吐升而单用户速度降的方向。动态W可以改善接近EOS/尾部的浪费，但不能靠已有acceptance更高就解决每用户预算。按当前full-active cycle不变的条件式W/cycle，C8约7.655、C16约3.771 tokens/s；这是特定算法/实测周期的条件参考，绝非物理硬件上限或所有用户上限。

只作目标量级检查：以既有user p50和相同输出工作量计算，C2从14.610到15需要时间下降约2.60%；C8从5.346到15约64.36%；C16从2.943到15约80.38%。p50变化不能证明逐用户全部达标，token变化后必须重算。删除draft仅有约5–7%的条件理想收益，无法填平C8/C16缺口；verify占比约89%也不等于89%都能优化。动态W仅在现有受支持W集合和真实测得的有用token/总cycle关系中选一个明确策略，保持状态恢复/正确采样或声明近似质量，不新增W16、C网格或强制全接受。C8当前路径尚未归因前，动态W候选DEFER。

继续STOP Kimi INT8v1/v2盲调、GLM664 selectiveINT8重复、已测724 prefill pipeline改名重跑、M15窄矩阵、TP32 full93、空DeepEP与power2 padding。W4A8只有旧旁支布局/provider参考，TP16 localI192可满足%64而TP32 I96不满足，当前FP4 group32与per-output scale不同；未证明Hygon provider和质量，DEFER CPU/source，不启动新完整模型量化加载。上游r130快照显示5f91ad到main仅3提交、Core/RT/Ops/CCL gitlinks相同；该快照并非本审核新fetch。“更新Coredeps”需依实际受影响provider进行隔离最小更新，不全栈换代、不把adopt-modern分叉当直升。GLM当前PP2定制DSA不符合上游PP1普通paged图资格。

资源按可测任务分配：旧快照在途GLM16+Deep8=24卡，最新Deep1134已终结，不能继续写24卡busy；GLM16继续，Deep原8卡当前可用/归属须root核实。r134b组件16卡与独立K119B33卡分别由root确认资源，320总卡/40held不等于其余卡均空闲或已授权占用。并行推进的是r134b实际1HOST资格、K119B当前1K event source、767结果分析和1134本地数值诊断；有真实源码/输入才占卡。TP32 constructor是内存资格替补，非性能主线，full93静态58.906GiB+4GiB floor仅1.094GiB未知driver余量，保持UNKNOWN_STOP。Qwen工具调用资格可用现成1卡源码独立闭环，但不挤掉上述性能卡、不计大模型目标完成。禁止dummy busy，也不为提高利用率启动大矩阵。

物理峰值：exact SKU、对应精度/稀疏口径峰值、频率、HBM规格与有效互连上限尚未闭合，均UNKNOWN。不能把参考带宽、实测717 TOPS或一个大GEMM结果称为物理极限。模型阶段roofline只能在已知真实工作量F、搬运量B、拓扑后用max(F/Ppeak, B/BWpeak, 必要通信下界)作有前提的下界；当前缺Ppeak/BWpeak时不出伪精确数值，不能把重叠阶段相加。当前最有价值的是实测critical wall、可移除比例与SLA/per-card曲线。

质量总约束：前期基本合理、答案正确、有限/生命周期/边界有效；后期真实benchmark同数据模板、采样与预算比较配对质量及区间、解析/截断/覆盖率，不要求输出token相同，不引入GPU F64。全部服务卡、失败/排队/未完成必须列入正确口径。第一轮需要得到当前实输入归因，第二轮需要候选whole-model配对收益或明确停项；编译、链接、constructor通过都不能代替它。

## 证据与落实要求

本地已读：r/astra-review-118-current.md；r/InfiniLM-hygon-test/AGENTS.md及coordination/QUALITY_POLICY.zh-CN.md、ASTRA_REVIEW.zh-CN.md；r/closure-state-root-r9360-a.json（仅选key）；r/live-entries-root-r9357-b.json；r/Kimi-concurrency-bottleneck-analysis-r9286.json；v/Kimi-pool-abort-readonly-r133a.json；v/Kimi-prefill-pool-fixture-fix-private-r134b/{actual_submit.hpp,src/prefill_pool_component_r134.cpp,CPU-local-contract-r134.json,producer-source-r134.json}；v/glm-critical-observer-r131/{glm_sparse_observer_r131.hpp,glm_decoder_observer_r131.cpp,glm_text_observer_r131.cpp}；v/glm-critical-model-entry-r133b/perturbation-policy-r133.json；v/upstream-audit-r130/summary.json与report.md。r为项目目录，v为本报告目录。

本报告的ADOPT/STOP/DEFER是审核建议，root须逐项登记实际采纳/拒绝/暂缓、owner与理由；本报告未代替派发。最新1K路径按root增量证据条件采纳，actual新candidate成功、输入完整、计时/质量仍待原owner证明。118缺测prefill overlap应为null的修复在本次选读证据中未见实际结果，保持落实UNKNOWN，不声称已完成。

