# I192 W2 group32 INT8 可行性与去重（r9281，CPU ONLY）

结论：数学与gfx936指令支持成立，可提交一个很小的W2组件原型，但目前仅CPU_CONDITIONAL；没有编译、GPU、Slurm或模型性能/质量结论。优先等待root决定；不启动任何原型。符号映射不是新发现，r8912已记录，新增的是绑定当前f6 provider后对I192 W2六个独立group32的具体数据流、额外暂存和真实histogram成本。没有检索到此同精度/维度/scale/dataflow方案的已执行记录；该去重结论限本次定向历史文件。

## 实际源码、对象与DTK绑定

现有b4 native是 b4f916dcf32f3bafb7451c94d6a98746a0e6c084edf73aeb6e229275579dbb8d；当前provider.o是 f6ae35e870a8730c7f8e564d566503c6c31c1d253ea10758b5a2d3959b2f0c9d。通过已reviewed run_stdin_remote_r9046.py stdin读取固定文件，远端实际provider源码SHA eb51cb12351b947ef2f2940ece880919ae0c5276a0292c8866d2e40693733095 与本地 Kimi-MoE-batch-profile-private-r9125/kimi_i192_batch_provider_r9125.cu 一致，actual provider.d所有依赖0缺失/0变化。回执 Kimi-INT8-current-source-readonly-r9281.json 保留完整MD与原文。没有使用其他workingcopy推断实际源。

已通过CPU700的 hardware_dumma_r8403.hip 使用 du::dumma::DUFragment<matrix_a,16,16,32,signed char,row_major>、matrix_b col_major、int accumulator 与 du_mma_sync；其原ISA明确有 v_mmac_i32_16x16x32_i8，底层gfx936 builtin为 __builtin_amdgcn_mmac_i32_16x16x32i8。CPU-build SHA c86433f011cd6f14dba622c91f901e014654a11f1f97ad6d72dec75bd8808e8c，原DSO c82d58eb54b7b3b74810bfacb2dc9afc8945749bce8b4f914333578773d5190b，汇编 bee199ebfb0772f4a214e85258b9fb324c8f84ad6d902711df9cf06df8b37866；当前DTK du_mma.h/hpp哈希仍与该构建相同（完整值见JSON）。integer-arithmetic-qualified-r8410.json给出原GPU702/8卡/96case独立CPU核验；这仅证明该INT8原语可用，不证明本W2数据布局已正确。r8804是vector FP32 ISA资格，不是INT8证据。matrix_hardware_roof_r4931.py测旧I384矩阵形状，不是本I192成本。

bw1000-hardware-roofline-r1311.json的717.923TOPS是历史大INT8 GEMM经验值，不是物理峰值。当前SKU/时钟对应物理INT8峰值UNKNOWN；不能从FP16字段换算，也不能从寄存器MMAC峰推模型TPS。无GPUF64。

## 数学与边界

令v(c)为E2M1数值，qW(c)=2v(c)，则qW∈{0,±1,±2,±3,±4,±6,±8,±12}，W=0.5*qW*2^(sW-127)。原native读取相邻低/高nibble，不能沿用Marlin blocked uint32的字节顺序。producer用Fraction遍历16个code×256个scale验证有理数恒等；数值零保持，负零bit不保持。此符号恒等不能证明极端scale的FP32 overflow/subnormal或scale255行为；实际组件必须对照实际ldexpf实现并保持finite gate，本次不读权重payload。

对每条route r的每个g∈[0,5]，sA=maxabs(A[r,g*32:g*32+32])/127（全零块约定sA=1，qA全0），qA=clamp(round_to_nearest_even(A/sA),-127,127)。输出近似为

Y[r,n] = Σg=0..5 { float(Σk=0..31 qA[r,g,k]*qW[e,n,g,k]) * (0.5*sA[r,g]*2^(sW[e,n,g]-127)) }。

每块I32点积绝对值≤32*127*12=48768，INT32安全且转FP32精确。六块必须分别清零I32并scale，再FP32累加；不能累满K192再统一scale。route weight仍在最后按原route0..15升序FP32归并，最后一次BF16store。weight数值精确不等于相同FP32舍入；activation额外INT8量化有误差，必须实际QA。producer的zero/mixed/outlier合成误差在量化理论界内，仅是公式/实现自检，不能替代任务benchmark。

单个线性signedINT4码不能精确容纳该码本：.5与6比值12而signed4最大正数7。双平面分解/重新量化是不同方法，不在本候选内；禁止直接nibble解释为INT4。

## 唯一候选：当前W2 grouped A8、W8 MMAC，W13不变

沿用当前counts/routes，W13完成后在同Runtime.stream把jobs计数清零并调用buildJobs<16>，可复用原jobExpert/jobStart/jobs数组（原W13后不再消费它们）。activation一次quant到原route索引R×192 INT8及R×6 FP32 sA；不增加CPU count读回或同步。每wave负责expert的16条route×16个相邻输出channel，沿K的六个32块运行：gather量化A到512B LDS；从原packed W2 on-the-fly低/高nibble展开signed8到512B LDS；du_mma_sync清零int fragment后只计算该块；用du_store_matrix_sync到1KiB LDS临时区域，每lane读回对应输出做I32→FP32、sA及sW乘法后累加；6块完后把有效行写FP32 route partial [R,3584]。新reduce kernel保持原ascending16route权重归并与BF16 store。invalid expert不进入counts，归并仍检查ids，避免读取未初始化invalid route partial。

原模型W2 bank不展开、原W13/activation/SITU/router/shared/TP网络不改。FP32 partial、量化和scale暂存使用实际Runtime分配/原owner生命周期，重复调用与canary必须覆盖。额外全局暂存仅R*(3584*4+192+24)，另CTA约2KiB LDS（A512+B512+I32tmp1024，不含编译器资源）。若frag布局直接读寄存器可减少LDS，但这属于后续有源码依据的实现，不能预先删掉当前保守成本。

## 全链成本：真实历史owner0（所有16owner完整值见JSON）

|T|routes|unique E|max rows/E|J16|physical rows|M利用率|新增全局暂存|
|---:|---:|---:|---:|---:|---:|---:|---:|
|8|128|33|4|33|528|24.24%|1,862,656B（1.776MiB）|
|64|1024|33|32|65|1040|98.46%|14,901,248B（14.211MiB）|

T8是4.125× M padding，T64为1.015625×。T64由历史16行重复4遍，33个expert高复用并非16个真实不同用户分布，更非在线流量。若真实T64 routes分散到614expert且每expert≤16，physicalrows≈9824、M利用仅10.42%，不允许复用本T64漂亮数字预测该分布。

T8/T64分别44,352/87,360次wave MMAC、0.727/1.431G padded整数OP、22.71/44.73M nibble展开值、11.35/22.36M逐块输出scale元素。朴素API路径的I32 LDS写回再读共90.83/178.91MB；A/B LDS staging另45.42/89.46MB。所有为dataflow计数，非实测流量。

原W2 per-route逻辑packed+scale读取46.79/374.34MB；按本expert16分块共享后为12.06/23.76MB，改善来自route复用而不是INT8字节存储本身。原W2pairCTA已经共享相邻channel的A，本候选仍需额外quant、gather、padding、6次I32转换/scale、partial和归并。partial单次写+读最少3.67/29.36MB，quant读取BF16 0.049/0.393MB、写qA+sA 0.0276/0.221MB；A跨224个Ntile重复gather仍有22.71/44.73MB逻辑读（含padding；cache可降低HBM，但不能当0）。jobs memset/build、所有kernel launch、guard、原route归并、Runtime分配/复用和完整TP MoE drain均须在wholeforward计时内。不能只报告MMAC kernel时长。

## 历史实际负例及新增区别

- integer-full-cost-reuse-r8425：r4966随机W13 K3584N768/W2 K384N3584，M<32 pad32，Python/Torch perstep转换/scale/pad/quant/dequant全部收费；M16 W2 INT8 0.434ms vs BF16 0.0687ms，整体pair慢1.328×。这是必须正视的额外开销负例。本候选I192/原MXFP4 exact W码/group32 scale/原route共享/单privateHIP流程，无整bank INT8和Python逐expert循环；差异足以提出一次可证伪小组件，但绝非保证收益。
- r856：I384 W13解码BF16 LDS MMAC，真实663早期组合快1.20×、晚期s383仅0.60×，force LDS后期负；r902直接fragment/pitch/hybrid不得重复。本候选W2/INT8/group32，但同样面临低M利用，因此真实低复用控制是必需项。
- corrected WFP4 r929：实际TP16 I192、FP32 W13 MMA/原SITU/W2，compound event均值1.3877→1.4456ms，慢4.17%；禁止借“FP4矩阵”标签重跑。它不覆盖本W2 A8量化和group32 I32路径。
- r943 FDOT2：I384 W13含halfprepare/grouping/fallback/原W2，actual663 s1组合0.698×、s63 0.723×，随机T64约0.857×。精度/维度/运算不同，不能直接否定新W2，但说明conversion成本必须收费。
- cold BF16 r781：T16高复用21expert快约1.139×，seed229expert仅0.396×，248.5MB explicit scratch外加vendor约75.5MB；本候选无bank cache，但同样不得只取高复用结果。
- EP2 r868：TP16 I192→TP8×EP2 I384，wholeforward latency +10.05%；cyclic r889 +3.85%且漂移，无稳定收益。本候选不改expert placement、TP/EP或网络，不重做这两条路线。
- 现r9280 W2pairCTA已经组件通过（T8约+5.14%、T64约+11.70%），root正在33卡模型验证。本候选如继续，OFF应选当时已通过的最佳W2pair版本，不能用旧f6作唯一基线制造收益；W13/objects/headers仍须继承实际sealed MD逐项绑定。

## 证伪卡与STOP

root若允许下一步，先只做一个privateHIP TU和一个wrapper/opt-in，在当前I192已sealed实际依赖编译，检查ISA确有i8 MMAC且无F64，API原语/fragment布局用小CPU期望矩阵核对（这一步本次未执行）。若需要整模型INT8 bank、另一推理框架或改共享运行依赖，STOP。

组件只做一组T8/T64原有histogram加一组低复用（明确constructed并保持每token top16唯一）的当前完整MoE ABBA，原weights只加载既有单层组件一次，不为本报告加载31min大模型。保存finite/OOB/canary、重复调用、relative RMS/maxabs作诊断并收费所有新增kernel；wholeforward按每rep maxowner再arm median。若仅高复用通过、T8或低复用>3%回退，则默认OFF/PARK，除非root接受明确限定的dispatch；不加参数扫网格。若均≥5% wholeforward收益且漂移不能解释，才申请挂搭下一次已必要模型加载作同采样GSM8K固定32题筛查，后续扩大正式benchmark比较任务得分、配对区间、截断、EOS/解析与完整服务延迟。token无需一致；不引入GPUF64。组件通过不等于模型收益、不等于质量无掉点。

当前动作：4096个符号恒等+3个合成误差界自检+现有histogram静态成本已完成，CPU producer和证据SHA已交root；CPU/GPU原型均等待root决定。
