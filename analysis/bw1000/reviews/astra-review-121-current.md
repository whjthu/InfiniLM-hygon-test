# Astra review 121：P0先收口，Kimi性能归因排在同组资源的下一项

结论：保留用户已明确的三真实模型P0优先顺序。Kimi144d先进行一次有界、自动发起客户端并自动收尾的真实P0；其正常终结或明确停止后，同380的32target卡转Kimi C8profile136b，另核371 n03一draft卡。GLM与Deep P0可并行。>90分钟无Model gain要求复核路线和消除无人发请求的空耗，不授权静默把用户P0降级。原prefill/decode/total/card及逐用户>15目标均未完成。

本审核只读本地源码/证据并写报告，无SSH、Git、编译、GPU、启动、改源。已重读当前AGENTS/ASTRA_REVIEW与review120。首判已发root；最初发现的GLM143c旧客户端引用已由原owner在审核期间修复，末次复核client147、supervisor pin、新manifest均匹配；当前未发现额外确定的正常启动阻断。独立预检不等本报告。

## 120落实与真实终态

| 路线 | 最新事实 | 结论 |
|---|---|---|
| Common C / R120-2 | bridge 0da5c247与server 3eacb768加入每循环绝对deadline、首次TIMEOUT/ERROR保留、HTTP504、SSE timeout；finish仍由owner reset后触发 | 只读静态收口；不重新造D或扩展协议框架。CPU fakeowner通过仅是控制逻辑证据 |
| Kimi / R120-1 | 144d KimiNativeEngineBridge在pre-SSE与enqueue共同限制output1..512，官方完整prompt1..512并检vocab；512+512可接受 | 确定输入缺口已源码修复；CPU合同不等于真实Native P0通过 |
| GLM771 | 实际FAILED124，service-ready但无客户端请求，postempty，未正常qualified | 不是数学/模型负结果，也不是服务通过；根因工作流未自驱。143c新增同机CPU subprocess自客户端和finally stop |
| Kimi432 | 实际FAILED1，ready、0 requests；退出时严格检查lazy首Ops affinity变化0导致assert，postempty | 不是forward失败或用户速度结果。144d区分idle观察与active严格检查并新增CPU keeper自客户端 |
| Deep1145 | GPU COMPLETED0、normal/postempty；10个主case中9通过，真实HTTP四工具闭环通过，仅SSE workflow旧client header检查失败 | 已有真实模型/HTTP工具进展；整体P0尚未通过，不重写旧结果为成功 |

source snapshot为services-observation-root-r9394-a.json，旧watch的follow/running文字不能覆盖终结行。root报告05:50–10:11 UTC额度中断；这段不能记成持续模型性能计算。GLM已ready却无人发请求说明下一运行必须自带客户端和stop，不再依赖协调者手动跟进。新whole-model配对性能收益仍0；这些服务失败不是新增性能优化波。

Deep147修复仅将HTTP header字段名查找改为大小写不敏感，仍严格要求text/event-stream、完整[DONE]/finish/usage、合法工具与真实执行。CPU-actual-wire-contract-r147.json复用了1145实际首条SSE wire SHA85cac56b，解析出真实read_file(input.json)、finish tool_calls、309prompt+54completion=363tokens。旧响应header原字节未记录：只能说旧代码对lowercase header可复现失败、实际body完整可解析，不能断言某个未保存header字节，也不能据第一响应追认后续四步SSE全部成功。新live SSE闭环仍是待办。

## 冻结源码的有界检查

本地逐个SHA核对：Kimi144d manifest f083f3ba含96个被引用文件（加manifest共97）全部匹配；Deep143d source88a18786含22个引用文件（加manifest共23）全部匹配；controller394cf118五个引用文件匹配；GLM143c旧manifest fc7da955的37个引用文件匹配，但client仍feb872dd。Kimi与Deep实际client均为147的41dbe048d14ec8f023bd3448426d6e87931f094e5919db6ee72957e0cdf1d77e。源码匹配不是远端构建/运行资格。

本次唯一确定启动前项已修复：最初GLM glm_auto_client_r143c.py的CLIENT_SHA256和包内client均为feb872dd。原owner随后同步实际147文件与pin并重封；本审核重新逐SHA核对新37文件无差异。最终CPUmanifest9d50ce5c00db58eeb36b66f8edf675af16e957088412537f7d205e1e23b7e91d，GPUmanifestebf1fd5ac63c9f65b16809330f6ca9d78f17108059c4d925c50524a36d761444，producer63b8583d74ac8f8a6e2a1a4d053d1c84eef517714930cfefd62f92168c6120aa，supervisor2f2485899a07ae2d00288045d867318a8f89ae4df8c434bbe20c99bb9a3cda4d；client精确41dbe048。旧fc7da955/4f6c5cca不可启动。owner报告4项CPU编排合同重跑通过，本审核未执行；实际新CPUprepare/远端发布仍沿原流程。147 strict parser、tool执行和usage规则不变。

GLM自驱结构静态有效：HTTP线程不作Native调用，CPU子进程运行client；原owner保持forward、sync/reset/world barrier，再finish。supervisor在正常/失败路径写stop；emitter在WAIT continue之前检查子进程退出并补stop，覆盖supervisor preflight先失败的情况。没有创建第二模型。实际客户端一旦退出必须保留P0 passed真假，不能把服务正常stop当P0成功。

Kimi自驱结构静态有效：ResidentP0Supervisor绑定同actual handle、模型Native/卡数/context、唯一CPU client输出，明确无重启；ready剩余不足900秒则写stop而非空耗；正常客户端终结/期限结束都请求原服务正常stop。client最长2700秒、Native cleanup在其后，Slurm时间已格式化为HH:MM:SS。144d active有实际请求仍调用原strict affinity检查；0请求且0forward仅观测8owner changes0/empty mask，明确active资格false，不手工绑核、不凭idle宣称通过。真正P0必须有非零真实request/forward和active验证。取消后从未发生Ops的极端情况仍不属于本轮正常P0启动阻断；若触发原strict失败应如实处理，不能泛化放宽active规则。

Deep143d仅客户端147和独立命名空间变化；原ac937生成loop、官方DSML、Common C、原loader AST/finally不变。controller引用88a18786并要求实际新CPU receipt后才GPU；不回退到旧client、不因相同Native重编数百个对象。已通过HTTP工具和正常清理可复用为历史证据，新执行集中补足live SSE及其余原固定计划，不增加测试网格。

Common C边界检查：超时只记录transport取消意图，未提前设置_ended；owner完成reset后finish保留严重原因；非流式明确504；持续progress不能推迟总deadline。未发现新的确定正常启动阻断，不因理论上存在毫秒级deadline竞争而扩大修复。三份服务仍全文buffered；role包不是首内容token，internal token时钟不是客户端ITL。P0只可报告协议、真实工具结果、端到端wall；不能报告增量TTFT/ITL或用户TPS达标。原max_pending非严格原生队列容量限制继续DEFER，不加并发压测。

## 下一项资源与退出决定

**K121A，ADOPT，Kimi原P0 owner，380四node32卡，无draft。** 假设：144d自驱后可在真实Native上完成普通baseline与原P0工具计划，旧0请求失败不再发生。仅用冻结144d/147，保持1K、cap8、Native305726、原数学与reset。准备复用，排队/加载UNKNOWN（约28.5分钟仅旧参考）；ready后客户端应自动在keeper下一个约10秒轮询周期发起，首条实际请求最多给60秒启动证据窗口，不等待人工。无法提交/客户端立即失败则沿原stop退出并保留确切错误，不空挂至硬限。第一条baseline完整输出和基本合理性判定后跑原10项及条件项；actual client最长2700秒为硬边界，不是必须耗尽。无truncation、4GiB floor/normalclose不放宽。一次有界执行后成功就结束P0本轮；若失败，消费明确错误并停止此包，不重做原矩阵凑通过。整模型收益UNKNOWN。

**K121B，条件ADOPT为同组资源下一项，Kimi性能owner，38032target+另核371n03一draft。** 不与K121A重叠，不热切服务进程。复用已source-ready/published136b、Native305726、1K cap8/pool17/pages68、现成GSM完整输入及C8W8 flatT64。先实际CPU closure与唯一rootGO，事件池必须在Runtime创建前开启；真实verify两次warm后采一个自然T64，不补padding/强制EOS，缺测UNKNOWN。沿冻结OFF/ON/OFF方案、3%扰动上限，只定位当前verify大头；不是加速候选已通过。准备目标15分钟核现有包状态，queue/load UNKNOWN，附加观测目标<=10分钟；真实整cohort不为时限截短。唯一target加载后draft按已有capacity门槛启动，33卡口径。没有自然T64/扰动失格/质量容量问题则STOP，不退回历史单层fixture。若有最大可改区间才单patch配对，完整quality/吞吐目标另验。

若K121A出现新的明确阻断而15分钟最小修复仍不能提供可执行包，P0保持未完成，才把已空闲且已确认的380让给K121B以恢复性能因果证据；这不是将P0无限挂起，也不取消在途任务。若136b自身未通过实际CPU closure则两者都不占GPU假忙。P0自驱和固定停止条件使此次优先顺序不再依赖协调者始终在线。

**G121A，ADOPT，GLM原owner，376两node16卡。** 147同步完成后用143c自驱一次真实P0；排队/加载UNKNOWN，ready后60秒内必须出现客户端启动/首请求记录，否则原stop收尾。原640 context与96默认output、完整tools history；超限HTTP400并报告workflow未覆盖，不删history。原client每请求180秒、supervisor1800秒上限，不新增性能矩阵。成功需客户端真实结果、16owner reset/normalclose与post；如失败记录实际wire，下一步只处理具体阻断。GLM767 G119B继续STOP，无>=10%可改区间，不再以P0加载为理由插入稀疏profile/M15/INT8试验。

**D121A，ADOPT，Deep原owner，374一node8卡。** 复用143d/147完成最短live补验，原完整P0计划同一次加载，可复用历史9项不作“此前无真实结果”叙述。准备只需新source/overlay实际CPU闭合，queue/load UNKNOWN；首判是live SSE第一条合法tool call，再完成四步链和恢复，测量目标<=15分钟（原硬限保留）。SSE返回错误或不完整则保留wire、不执行半个tool；原模型和parser不改。成功仅P0协议与工具闭环，非benchmark不劣、增量SSE或吞吐提升。

## Deep数值与剩余目标

Deep1138原raw已完成state cases+4controls、normal close，加载370.26秒、总517.51秒，numerical_quality仍UNRESOLVED。ordinary repeat/matched T4 full accept与same-T4 rejected-future controls必须分别读；已有shape/ordinary差异不能单独证明rollback错误，也不能据某一rejected_future零差就证明所有自然任务质量无损。P0继续独立原ac937普通AR；新prefix/MTP路线DEFER，无MTP head，不冒称spec性能。

ADOPT：三真实模型一次有界P0、自客户端/自动stop、统一147、KimiP0终结后136b真实性能归因。STOP：GLM旧client包直接加载、ready后等待手工发请求、终态失败自动重提、P0 success当吞吐goal complete、GLM无候选新profile、Pool全层推广/TP32 full93/盲量化。DEFER：小Qwen、动态W、Deepprefix数值晋级、增量SSE和并发SLO。无需新增framework或门禁；只复用已有凭据及实际变更的最小CPU检查。

质量口径不变：早期基本合理/答案正确、有限及状态有效；后期真实benchmark同数据/模板/预算比较配对得分与区间，不要求输出token相同，不用GPU F64。prefill/decode/total/card、user>15与物理峰值边界仍待真实证据；exactSKU peak UNKNOWN，不拿717TOPS或参考带宽当物理上限。root须登记本报告ADOPT/STOP/DEFER与原owner落实，目标未完成。
