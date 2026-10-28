# Astra review 122

结论：恢复性能优先。三模型已各获得前审要求的一次有边界的 P0 机会；后续协议问题先用保留证据和 CPU 修复，不再自动触发大模型加载。新增整模型加速仍为 **0**；Kimi 本轮得到的是合格的当前输入瓶颈证据。

## 前审落实与实际指标

- Kimi 438 P0 10/10、恢复442/32 Native 正常退出为 root 最新证据；原失败不抹除。445/592 的 C8W8 OFF/ON/OFF 共12 case，四个真实 T64 窗口覆盖32 owner，最大 observer 扰动2.895%。GSM32每次正确/EOS，三次重复不能算96道独立题。buffered SSE 不产生真实增量 TTFT/ITL。
- pooled decode 采用总有效 token/总 decode window：OFF前38.9147、OFF后38.8838 tok/s，33卡分别1.1792、1.1783 tok/s/card；user p50为5.2096/5.2229，0/32用户超过15。不同 cohort 的33–44 tok/s不是新的集群记录。首次 prefill 冷热差也不是优化收益。
- GLM778完成20次warm call后在A1被60s控制器取消，A1真实进展未知；不算性能轮次。warm serial/deferred 的48/32秒不可当收益。
- Deep1152是实验F100普通重复数值诊断；partial/full形状混淆，未证明rollback错误。root报告生产ac937/1156仍9/10，第二tool在96截断；不是数学回归或模型加速证据。

## Kimi的可用边界

|128 owner-window记录的均值|ms|
|---|---:|
|stage0 whole/local|429.798|
|stage1 whole|877.480|
|stage1 incoming（model.begin→decoder.entry）|429.468|
|stage1扣除incoming后的本地服务|448.012|
|两段本地服务需求之和|877.810|
|两段MoE routed/shared + TPwait之和|552.743|

MoE占本地服务需求62.97%，KDA约22.61%，MLA约7.70%；这些是stream依赖区间，不是纯GPU忙碌时间。**不能把stage0与stage1 whole相加**，不能将stage1 incoming再次归为可优化MoE。现有profile还不能分清router、packing、GEMM、专家偏斜、CCL。

以下只是假设边界，非实测收益或物理峰值：沿用此前verify占cycle 89%且工作/接受率不变，MoE快2倍对应cycle约1.389倍；MoE零成本约2.275倍，p50只投影到约11.87 tok/s/user。89%必须在当前cycle重核，不能由这128行独自证明。超过15要求当前p50至少约2.87倍提升。

平衡两段PP的稳态verify容量上界约1.959倍，但不等于单请求时延减半。cap8拆成两个C4W8/T32时，应实测t0(32)、t1(32)；填充/排空时间为 t0+t1+max(t0,t1)+反馈，不能假设T32成本是T64的一半。精确SKU物理峰值仍UNKNOWN。

## 下一轮只保留两张Kimi卡

**K122A，先执行：真实T64 MoE分段，随后只做一个证据支持的改动。** 原Kimi性能owner复用136b/Native305726和原GSM32，20分钟内给出最小稀疏事件源：router/dispatch-pack、routed matmul/activation、shared、TP collective，并保留每rank/专家token分布。事件预建、原stream、不新增测量内host sync。一次热身后OFF/ON/OFF四个cohort真实T64/all32，observer≤3%；当前对象和输入不能替换为历史fixture。组件重放必须来自这次真实输入且明确仅组件证据。GEMM主导才改一个形状/融合；packing主导才合并搬运；均衡rank仍TPwait主导才试通信调度；偏斜明显先改分组。没有占串行关键链≥10%的可改子段则停止宽泛MoE路线。source不在20分钟内就绪则停止本轮扩张。排队/加载时间UNKNOWN，分别记录；支持的候选用同一加载ABBA，whole-model收益须>3%且超过配对漂移，质量/内存floor/退出不退步。

**K122B，并行准备但不可直接启动：cap8内C4W8×2的PP重叠。** 原stateful Native owner只用15分钟做可行性审计。现有batch_native每次arm改写一个taps epoch，finish先control.collect再harvest；cohort等call后才commit/replay。普通两次call仍串行，直接推迟finish可能覆盖第一组tap/事务，不能照搬GLM双logits helper。必须先给出两组独立capture/row事务、原caller、固定collective/feedback顺序、各自commit/replay和无重叠逻辑行。做不到即STOP启动。安全源码成立后，同一加载先测T64串行与T32×2串行的形状代价，再做T32串行/重叠ABBA；以真实原32请求的有用token、每用户时间线、EOS/基本质量和正常退出判定。新T32形状控制不等于重复旧整模型矩阵。没有超过3%且高于漂移的whole-model收益就停。

W16暂缓：当前明确maxflat64、每行width≤8；C8W16是T128，C4W16虽T64也超width合同并改变并发。必须先有接受token/cycle与verify+draft+replay成本证据，不能以长窗口“理论多接受”取代性能实验。

## GLM与Deep分配

**G122A ADOPT：r153实际CPU准备通过后，只跑一次修正后的arrival32/maxlive32/M16×2 ABBA。** 审核重算producer全部23项hash无差异：producer b334fa62…，manifest d48fe082…；相对148归一namespace后，probe唯一变化是每次成功WORLD command exchange后、Native前写atomic progress，原60s guard保留。helper仍20482dc…，两组logits各自持有，最多两个pending、原顺序反馈；all16审计验证command序列、实际Native覆盖、行ledger与退出。未发现新增确定启动阻断。无Native/依赖重编译；n14/n15独立16卡可与Kimi源码准备并行。

仍需区分：新心跳解决“可能稀疏上报误判”，没有证明778 A1无hang。若r153单command仍超60s，保留seq/op/Native入口证据后停，不能循环放宽timer或自动重播。保留原2warm+ABBA、2400s测量预算、prefix8/M15OFF/observerOFF/row2ON，warm缩短输入不计正式收益。实际两micro覆盖、完整质量和>3%配对whole-model增益才升级；否则停止此路线。767没有≥10%可改interval的结论继续有效。

Deep暂缓额外GPU：先用保留wire定位96截断的工具序列，再做匹配形状的普通重复分析；不放宽任意数值threshold、不拿token完全相同当最终质量门槛。没有MTPhead仍不能声称spec性能。

root应逐卡登记ADOPT/STOP/DEFER并立即派发有时限源码工作，不等发布报告。停止无修复的P0加载循环、盲量化/W16/grid、TP32全93、历史fixture冒充当前输入及dummybusy。主要硬件吞吐目标尚未完成。

