# 三模型分区调优与集成规则 v1

用户授权三位原模型agent分别负责GLM、DeepSeek、Kimi，root统筹框架接口、整合和合并。2026-11-06实际核实三位均errored usage limit；任务归属保留，消息已送原agent，不重启/换agent绕过额度，不声称三个执行者在运行。root直接继续可执行原scope。

## 资源和工作队列

| 任务线 | 固定区域 | 算力 | 第一阻塞/实际下一步 |
|---|---|---|---|
| GLM | parent910371,n03/07/15/16 | 32卡 | 完整16owner Native入口首验证；成功后16组runtime参数 |
| DeepSeek | parent910374,n01/04/05/07；910376,n12..15 | 64卡 | r482恢复35缺TU真实argv/依赖；闭合ABI后18组参数 |
| Kimi | parent910380,n08/09/10/12 | 32卡 | 12组tiling源码成本筛选；首选Q32/K6144和Q32/K12288，核SDK/数值后GPU |
| 控制 | parent910407,n20 | 8卡对应节点 | CPU构建/keeper；不把CPU工作当GPU busy |

每区保持至少2个READY任务，另存NEEDS_SOURCE/BLOCKED原因。没有READY时立即推进最低成本阻塞或转替补，不重复旧负族/原模型加载。READY数量、节点独占预约、请求卡数和实测busy分别报告。默认整节点独占避免性能实验互扰；单卡组件仅使用一张卡时，节点剩余卡不能计为忙。多个组件若要同节点装箱，另写PCI/进程/算子干扰隔离的精确claim，不直接共用旧fresh合同。

跨区借用由root发布新具体claim，不再询问用户；先核原实际句柄与活跃step去重。静态分区可变，未知执行不能因lease过期判空闲。目标是全136卡实测busy>80%，不是allocation或模型加载比例。五区域同步sysfs9点40秒看全卡分母；运行器内部不新增常驻GPU poller。20:53和本次22:07UTC样本都0%，目标未达。

## 参数空间和搜索预算

`bw1000-campaign-v1.json`含46个调优提案及1个GLM完整入口前置运行，不表示47个已就绪作业。每个family是一个明确假设；新方法扩展family，不能把所有未经证据的参数一次笛卡尔放大。已封负家族进入去重表。

GLM：prefill chunk512/1024/2048/4096 × owner cores2/4 × compact/local，共16组；runtime适配器需先实现并冻结。DeepSeek：constant IDs OFF/ON × cores2/4/8 × prefill1024/2048/4096，共18组；classsize闭包未完成前不编译或上卡。Kimi：Q8/16/32 × K1536/3072/6144/12288，共12组；保持完整32768、逻辑512和原4GiB floor。

先源码/静态成本筛选→必要CPU闭包→实际组件数值→固定整模型负载→宽质量→服务曲线。每个family组件最多2个首波幸存候选。Q/K成本只是FA调用和显式临时bytes，SDK workspace/driver峰未知，不能按FA次数宣称加速。当前Q32/K6144、12288的额外显式buffer约6.75、20.25MiB，均不能用组件48MiB观察余量证明整模型fit。

按预注册3、9、27个完整配对测量单位分rung，最多保留top1/3。排序的 `--rung 0/1/2` 分别核对 3/9/27 对，默认第 0 轮，配对数不符的候选保留在拒绝列表；不得截短题、减少context、改变生成预算来缩rung。每次先正常reset/quiescent；同Native/权重/资源、支持切换的runtime参数尽量同一次加载跑多个arm。需要换编译实现/数学的配置另冻结库，不能热改活跃共享依赖。基线采用ABBA/交错、记录warm顺序/JIT；不挑cold baseline美化。缓存旧证据用于资格和去重，新配对如需基线只随新对照一起跑，不独立重复旧套件。

晋级依据预热后同口径逐用户rate配对、质量、显存floor和生命周期；报告TTFT/ITL P95/P99、吞吐/全部卡、goodput、load/queue/测量/质量耗时。bootstrap只作筛选不代表正式NI；正式NI margin目前未预设，不能捏造。质量32题仅screen，后续沿原固定256题/臂及完整分母。component不能与model/service同排名。未知、失败、拒绝、取消、cap、未完成全部保留，失败不因后见文件可见洗PASS。

## 可执行队列与一次性派发

`tools/tuning_queue.py`使用stdlib SQLite，在忽略的`local-only/tuning/queue.sqlite3`保存候选、运行、事件和预约；不把运行游标/心跳提交Git。expand/seed仅登记，不能启动模型。source/CPU仍由原有CPU-only launch adapter执行；GPU队列只接component/model/quality/service。

prepare必须有root已读精确source/deps/claim/launcher及输入输出绑定、必要前提原证据、已ordinarypub精确commit且在origin/bw1000-experiments祖先上、local和published bytes SHA一致、原floor/预算/区域不漂移。prepare不运行对方代码。性能ranking使用完整comparison身份（模型、硬件、精度、scope、权重、workload、metric），不混CPU/组件/整模型。

plan读取最多60秒的完整17节点/136PCI空闲快照，只选择原allowed_nodes中KFD/PID/端口/活跃step均明确干净的节点；缺数据不是空闲。reserve在BEGIN IMMEDIATE事务中重算计划，两个调度者只能一位成功。输出是root交给对应已冻结launch_once的调度包，工具本身不执行任意shell/launcher，也不是已经部署的常驻自动Slurm服务。

root随后只调用该模型已审launch_once一次，把实际requestID/intent/HOST/PID/startticks/label记入submitted；连接未知记UNKNOWN，资源继续保留，读原request/intent而不replay。即使本地launcher退出也不等于远端终态。finish必须精确原身份、真实terminal和独立post clean才能释放预约；保存keeperFAIL、分母，失败不能晋级。不同rung是预注册独立新运行对象；同candidate/stage/rung不能重播，下一rung须实际上一终态promote。

常用命令（在G运行固定Python -B -X utf8）：

```text
tools/tuning_queue.py seed coordination/tuning/bw1000-campaign-v1.json
tools/tuning_queue.py prepare exact-run-binding.json
tools/tuning_queue.py plan fresh-fleet-idle.json
tools/tuning_queue.py reserve chosen-plan.json fresh-fleet-idle.json
tools/tuning_queue.py submitted RUN original-handle.json
tools/tuning_queue.py unknown RUN original-handle.json
tools/tuning_queue.py finish RUN original-terminal-and-post.json
tools/tuning_queue.py rank same-scope-paired-results.json --rung 0
```

## root集成

三位agent只写各模型明确源码/小结果目录或隔离staging，不写root status、不并发操作共享Git。交接须明确base、patch/sourceSHA、依赖、实际原件、默认开关、原数学和回退。root先核局部CABI/依赖、适用硬件和真实收益，把通用实现落到固定InfiniLM相应框架位置；模型特有实现保留适配层。跨模型改公共Runtime/SDK时三模型相关回归必须闭合，不能以一个模型组件成功默认另外两者通过。

root串行集成、doublewhjthu/普通SSH推bw1000-experiments；默认OFF候选通过同口径模型质量/性能再决定启用，负候选保持证据/可复现版本。禁止自动合main/force/API。每3真因果轮/2轮无收益/90分钟无可测/严重数值/重复入口错触发已有Astra147一次；当前208额度不可用待审，无假审核在途，其它独立线继续。

## 当前实现资格

20项本地队列不变量测试通过，包括并发预约、未知不重播、原身份、终态post、失败分母、不同scope不混排与已发布包准备。新GLM loader另2项CPU录制通过，只证明覆盖/强引用/64位地址，不是Native数值。Kimi12组静态成本已真实运行；0新GPU性能资格。三agent仍额度中断；当前没有持续自动填满集群的执行服务，>80%未达，root继续冻结就绪包后实际派发。

GLM首次v1同容器HOSTpreflight90秒失败，未GPU。唯一staged owned-PGID诊断实际57.75秒通过且groupgone，Torch导入45.8秒；不能反推旧超时具体阶段。完整入口v2复用该实际环境证据，Model/math/floor/deadline原样，新命名空间先publish再preflight/fresh一次。筛选另要求完整planned分母；增加拒绝/取消或质量损失不得由快速完成子集自动晋级。

首次实际队列接入在prepare后、reserve/launcher之前拒绝了快照：旧freshness错误地相减Windows与集群wall时钟。新规则以原请求的本地monotonic往返上界+一次接收回执衡量60秒新鲜度，保存原request和payloadSHA，禁止重读旧文件重新盖章；本机reboot/大clockstep使回执失效。remote wall只描述，永不作跨HOST相减。原run仍READY，不能重新prepare或假称GPU失败。

## 895实际终态与终态记账修复（2026-11-06）

最新campaign-v4为51有效spec，SQLite52含历史、5实际GPU派发均已TERMINAL、READY0；每区至少2READY目标未达到。895完整保留1249/560输入、4096输出预算、32768容量和8GiB门槛，16owner完成117060参数加载、每rank389次全78层forward/4reset/正常销毁；两题自然EOS/答案正确。后端原时钟约2.92/2.89tok/s，0>15，不是宽质量或客户端SLO。原GPU895C0/1637s，但keeper1108FAIL1/1639s：srun0后首次sacct仍RUNNING，旧控制器看到有row便退出可见性等待。后见GPU成功不清洗keeper失败。

共享slurm_accounting.py及独立terminal-collection-fix-v1/controller.py overlay只在实际终态state才结束观察，保留缺行/RUNNING/terminal全部记录；沿原keeper单epoch，最多30s有界重读，并留下post预算，不另起poller、不重复launch。4个CPU recording回归覆盖895竞态、FAILED不改写、deadline UNKNOWN和缺行；未声称新collector已经真实Slurm验证。895冻结源码不变，不为收集器单独重加载模型。后续新完整包必须纳入该overlay、exact MF/deps/claim和原输入预算后再派发。

895两题只解除了完整入口容量/生命周期阻塞。下一优先同一次checkpoint加载里的真实forward成本/预注册3完整配对单位runtime配置，后续固定32题screen和原256题质量；按完整分母决定晋级，不由2题正确或约90MiB采样余量直接晋级。原decode时钟约94%–97%位于fullforward包围区间；其中含finite/floor读回，不能冒充独占GPU时间，剩余也不能全归日志。Deep20复用对象原5803 recipe/quoteorigin已恢复，但全50freshpreprocess/ABI仍待；Kimi更大K网格已负结论停止，下一单张V-only/pagedstride成本卡先源码支持和显存预算。

## r473派发前时限一致性与准备记录退役

同次checkpoint加载的r473为两完整case各自A4核/B2核预热(4请求)，然后AB/BA/AB三组完整配对(6请求)，总计划10；原完整输入/4096预算/32768容量/8GiB门槛，Native1101不变。现有六scalar TLS getter仅复制已计算的host/stream incoming/layers/outgoing窗口，不额外event/forward/sync，不叫exclusiveGPUtime。是两unique题重复的性能探针，不是32题质量或客户端SLO。

v1在fresh/reserve/GPU前发现keeper shell14980s仍旧，而新GPU18000/keeper18600。原source/publication/HOST metadata资格保留，未提交GPU；新不可变paired-ownercores-v2 keeper18580s+kill10s在18600内且大于18000GPU+post。跨shared.py/claim/keeper.sh/inside.sh/controller/launch/control的check_deadlines_CPU.py已集成HOSTpreflight，同检查实际拒绝v1/通过v2。不能热改已发布v1或因本地脚本返回而重播。

队列增加retire-ready：仅READY、nodes[]、handleNone时可凭原因与精确证据SHA转RETIRED_PRELAUNCH。它不是GPU实验终态/性能轮；RESERVED/SUBMITTED/UNKNOWN必须按原句柄reconcile，不能用退役释放。旧candidate/stage/rung仍不可重复prepare，修复使用新source身份。23队列不变量通过，包括退役不派发、未知/已预约不能退役、理由/证据必须齐。源/CPU错误不计性能，原审阅208待额度不假完成。
