# K122B / r154 — STOP 直接双 microbatch 重叠

结论：当前 Native305 与冻结136b不能仅靠Python调度，安全保留两组C4W8 verify。在本轮15分钟可行性门内停止扩张重叠源码。**本目录没有GPU实验、没有Native更改、没有新的吞吐结果。**

## 实际代码的三个覆盖点

1. `136b/runtime/batch_native_r9128.py:32–65`：`arm()`递增单`self.epoch`并重新`taps_begin`；`finish()`先WORLD `control.collect(selected)`、callback，然后才harvest。两次普通`call()`仍全局串行；直接延后`finish()`会让第二arm覆盖第一组特征。taps C++为进程单份`metadata/records`，begin直接`records.clear()`。
2. `Kimi-speculative-batch-private-r9122c/pair_probe.cpp:288–295`：Context只有单`batch_epoch_r9122/batch_requests_r9122`；forward末尾覆盖requests。commit要求logical数组逐项匹配**最新**requests。A=[1..4]、B=[5..8]即使KV行不重叠，先forwardA、forwardB、commitA仍失败。
3. `r9122c/batch_state_r9122.hpp:25–42`：每个owner Runtime只有单Cached；begin会清空前组捕获的layer tensors/scalars，并改epoch。state replay验证该单epoch与requests。只复制特征bytes不足以保留KDA部分接受回放所需的缓存。

这些源码来自本地保留的r9122c；Native305 receipt明确列出该候选pair_probe.o/worker.o/delta.o。CPU合同仅执行冻结Python方法和C++源锚点/行为反例，**未声称本地C++构建或ABI动态验证**。`CPU-observed-r154.json`记录实际SHA。

## 为什么本轮不直接扩写

最低修改范围包括：Native Context pending表按ticket/epoch保存requests；每个Runtime双独立回放capture；taps按ticket存取并显式释放；Python输出/feature/decision拥有独立生命周期；原owner线程单call；全rank一致的固定collective顺序和feedback顺序；每组各自commit/replay后才能重用行。错误/取消必须drain两个ticket，禁止先free后使用。原C ABI不能选择旧ticket，因此该工作涉及Native ABI与生命周期资格验证，超过本轮“无Native大改/无重载”的范围。未创建貌似可运行的overlap补丁。

成本估计按必要操作计，而不承诺尚无依据的完成时长：至少3处Native存储/生命周期改动（Context、Runtime replay、taps）及相应ticket C ABI，2处host调度/完成逻辑改动（Native wrapper、cohort）；至少1次Native重编译。CPU需要双pending、row冲突、旧ticket反馈、部分接受/全接受/EOS、取消/drain与collective顺序合同；GPU先跑一次有边界的生命周期/输出质量资格，再在同一加载中跑serial/overlap ABBA。两组T32回放缓存的总token量虽等于T64，不能因此承诺峰值内存不增加：两组输出、调度元数据与暂存张量生命周期可能同时驻留，须实测floor。工程工时、编译/排队/加载时间均UNKNOWN。该成本明显高于直接执行一个现有队列的形状对照，因此排在MoE当前输入细分之后。

## 可立即复用现有入口的最小形状对照

`prepare_shape_queue.py`从136b原始四组GSM32 OFFbefore队列生成纯数据命令，保留Native305、模型、TP16PP2、state、W8、量化、W2/I192、cap和问题，不改runtime。root可按现有上传和控制器流程采用；本agent没有提交。

- `first8-minimal/`：A1=C8；B1=C4前四再后四；B2=C4后四再前四；A2=C8。6个OFF throughput case，只覆盖原8个独立请求。随后C8与C4各一次OFF/ON/OFF，合计12个case和stop。
- `original32-optional/`：把四组原问题都执行上述ABBA；24个throughput case，另6个profile bracket case和stop。仅预算足够时采用，不能让它自动启动另一次加载。
- 每个命令是**独立closed cohort**，中间会`native.reset`、prefill、`begin_cohort`。两组C4不是同一cap8 live cohort，不能报告“C4×2流水线吞吐”。该计划只测形状成本与闭合小batch用户速度，尚无真实cap8双micro服务时间线。
- 每个cohort的prefill、decode分别统计。B吞吐按`两组有效decode tokens / 两组decode window之和`，不平均两个tok/s；仍须注明没有计入第二组到达排队，也没有跨cohort服务吞吐资格。
- profile各窗口必须实际flattened T32/T64及全部32owner，OFF/ON/OFF扰动≤3%，三次同问题只算4或8道独立题。各stage取local interval，stage1扣除`model.begin→decoder.entry`，不能重复计算incoming wait。
- 所有case用有用非EOS token和真实输出cap；输出合理性/EOS/数值正确性分别核验，不要求token逐位一致。记录draft、verify、commit、state-replay、feature/feedback时间以及接受token/cycle。
- 总预算先由root定；不足不续播剩余队列、不排第二次模型加载。初次仅采用first8计划，若当前模型已驻留，直接续入现有queue。

## 拿到T32测量后如何做去留判定

当前T64实测local需求`t0=429.798ms, t1=448.012ms, S64=877.810ms`。这些是依赖区间，非物理峰值。

设T32真实本地需求`a=t0(32), b=t1(32)`，`F`为本轮额外feedback/draft/replay等不能掩藏的需求；不要假设a/b为T64的一半。

- 两个T32 micro单次填充/排空的理想verify makespan下界是`a+b+max(a,b)`，尚未计feedback。若此值已经≥877.810ms，有限pair在verify层本身没有优势。平衡时需每stage<292.603ms才有可能。
- 零feedback稳态每两个micro的verify容量间隔下界是`2*max(a,b)`。若≥877.810ms，即使生命周期重构正确也无稳态verify容量优势。平衡时需每stage<438.905ms。**这条比有限pair更宽松，二者不可混用。**
- cap8仅允许两个C4同时在系统内；同组下一轮依赖上一轮输出、接受决定、回放与草稿，闭合系统还受`a+b+F`往返限制。原“平衡PP约1.96倍”不能直接套到拆小形状、同cap8与有feedback的服务。
- 任何最终whole-model收益必须在同一加载的安全原生双ticket实现上做serial/overlap ABBA，收益>3%且超过漂移，原GSM32完整质量、每用户时间线、显存floor和正常退出均合格。当前结果尚不支持这一阶段。

运行CPU合同与生成队列均使用bundled Python `-B -X utf8`；无SSH、无Git、无GPU、无Slurm操作。
