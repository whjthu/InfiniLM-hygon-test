# Astra 124：先把已有候选变成实际 GPU 结果

结论：立即完成 CPU503 已编译的 Kimi W13 两层组件；有净收益才支付一次完整模型加载，同次覆盖 C1 与 C8。Deep1168 是有效的真实 QA batch frontier，不能称算法优化或在线服务提升。GLM781 的 task +10.98%、decode CAPI +17.34%继续保留。当前最严重损失是结果消费和派发停顿，而不是缺报告。

本次只读本地证据，未执行 SSH、Git、作业启动或源码修改。复核了 AGENTS、ASTRA_REVIEW、review123及implementation123、Deep1168原结果/清理、CPU502/503回执、Kimi449子阶段及r155源码。

## 本轮已经得到什么

| 模型/工作 | 可采信结果 | 限制 |
|---|---|---|
| GLM781 | 16卡，task 14.262→15.827 tok/s；decode CAPI 21.670→25.427；用户p50约0.744→0.861 | 同32题31正确、32EOS；用户>15仍为0 |
| Deep1168 | 8卡，B32/16/8/4 common decode=36.815/20.566/11.989/6.712 tok/s；每卡4.602/2.571/1.499/0.839；用户p50=1.197/1.747/2.393/2.372 | 同64题均62正确、64EOS、0cap；顺序32/8/16/4、仅B4短warm；不能据此宣布优化winner/广泛质量不降 |
| Kimi449 | C8、width8、TP16×PP2+draft共33卡；profile OFF约39.2 tok/s、用户p50约5.24；L1/L47捕获区间W13+W2约86% | profile归因，非新收益；不是全层纯kernel份额 |
| CPU502/503 | Linux loader warm约1.65×、rotating约1.19×且合同通过；W13新候选编译48.17s、CPU合同通过 | 均不构成模型/GPU性能结果 |

Deep加载451.18s、四臂测量wall合计2110.15s。controller于03:37:10 UTC已COMPLETED0:0且post_empty；r9457-a采集时间13:44:39 UTC，消费滞后约10h07m。该时间差证明长延迟，不单凭两个快照声称整个期间全部节点空闲。恢复执行后的首要动作应是消费终态并发出已准备候选，不能再等待一轮心跳、Git窗口或审核完成。

## 下一轮实验卡（每模型最多两张）

**K124A，立即执行，Kimi原owner/root；n01+n04、TP16、16卡组件。** 假设W13归约的LDS/串行成本可减少。复用CPU503与r155：保持TT2/Tile4、grid640、FP4解码与FP32乘积，只改wave-first归约，LDS由16KiB降为256B；它不是新的token复用tile，且不能预先宣称occupancy提高。沿用L1/L47、historical-repeat和Gaussian两输入、4warm、3组OFF/ON/OFF及现有finite/NRMSE≤0.005/tail检查。已有≤600s GPU界，首个完整配对预计数分钟、队列UNKNOWN。使用full-MoE wall和W13双结果，不能只看kernel点数；现有≥5%且OFF漂移≤3%规则足以，不另加门。失败/无净收益则停此归约，转已有W2pair小候选，禁止再重测旧A8/TT1负路线。GPU状态未知不重播。

**K124B，K124A有效后，Kimi原owner；复用当前33卡服务资源，一次加载。** 假设该改动在真实单用户及多并发都有可见收益。同一加载保留开关，先C1短warm及完整自然EOS小组，再C8同32真实QA；各做OFF/ON/ON/OFF，沿用原speculation/质量规则，减少模型加载次数。C1结果单独报告，不能把C8的5.24当C1基线；组件只有T64覆盖，C1实际verify形状需在此顺带核实，小形状无利则保留原分支。利用原drain读取真实U/J、尾tile，不为histogram专门加载。报告客户端/现有emitter边界、逐请求first-to-last、TTFT、ITL P95/P99、总/每卡及达标吞吐。首个模型对照至少加载25–30min加测量；总约40–60min为工程估计，排队UNKNOWN。基本合理/正确后做真实benchmark无明显掉点，不要求token一致。若C1受益但C8退化，按形状分派；均无收益停整模推广。若A失败，B只搭下一次必要模型加载，避免无候选重载。

**D124A，可独立推进，Deep原owner；n05/8卡。** 假设B32先完成8个micro4 prefill造成长ITL，可用现有逐行past支持的最小调度改动改善。原结果first→second gap中位27.67s、最大56.46s（B8为4.49/8.85s）；源码明确先全部prefill再decode。同B32基线与“每个micro4后对ready行做一次decode，最多7次插入”配对，保持同64QA、自然EOS、EOS退休和token时间线，不将first时间后移。先CPU验证逐行past/新行加入，再同一加载warm后ABBA；预计准备10–20min、加载约8min、测量约18min，排队UNKNOWN。目的是降低长ITL；插入decode会推迟后到TTFT，且不能消除约124s总prefill，不能许诺15tok/s。若只是把长空隙搬家、质量回退或task损失不值得，则停止。复用B8作为延迟导向候选、B32作为吞吐导向候选，不再扩展四点网格。

**D124B，暂缓独立重载；搭载下一次必要Deep加载。** 加一个C1自然EOS真实QA基线及同实际形状的轻量已有profile，判断低并发的计算/通信/调度份额，才能决定kernel候选。B8与B4用户p50近似而吞吐约1.79×，不支持继续单纯减batch会达到15。沿用同题/同计时/完整EOS，不以B4代替C1。首个结果为加载后数分钟；当前缺可证伪kernel改动，收益UNKNOWN，无理由为此另开大矩阵。

**G124A，采纳现状，GLM原owner，无新GPU费用。** 保留781 deferON为已测负载基线；不要因为用户p50仅0.86抹去已得到的真实改善，也不要称目标已达成。

**G124B，次优替补，GLM原owner；独立n12–15/16卡资源。** r158 chunk128+prefer_decode是策略包，下一次必要加载做full/chunk128 ABBA，两臂deferON、同arrival32/自然EOS。若Kimi/Deep均因明确外部阻塞无法推进且该GLM资源确实空闲，可一次有界执行现成包，不以永久defer代替计算；否则不抢主线。仍报PREFIX税、TTFT/ITL和task；旧chunk32负结果不重跑。首结果时间由实际GLM加载决定，当前UNKNOWN。仅ITL改善而task下降时记Pareto取舍，不能通用替换基线。

## 硬件与Amdahl边界

同一计时边界下，若候选覆盖关键链比例f、该区间实际加速s且其余不变，S=1/(1-f+f/s)。Kimi当前W13是scalar FP32处理FP4，不能拿333TF BF16 dense峰值估算；真实HBM字节、算术峰值、资源占用上限仍UNKNOWN。

仅作条件敏感性：若445约63% MoE关键链份额与449稀疏层分布可代表同一真实负载，则W13约占整链0.32–0.33、W13+W2约0.54；这不是已测全层份额。f=0.33时区间1.2×/2×只对应整链1.058×/1.198×，即使零成本上限约1.49×；f=0.54、2×对应1.370×、零成本约2.17×。消掉全部63% MoE的条件极限也仅2.70×。不能把这些倍数直接乘用户p50或当物理硬件峰值；达到15需要C1/C8各自端到端证据，不能靠W13单项许诺。

loader：235M blocks约0.5s的CPU微测与1180–1408s Native安装差距巨大。99.77%是block数量份额，绝非wall份额。保留502候选、停专用Model加载；只在下一次必要加载加入已有边界的gather/state_dict/H2D时间和CPU/fault/IO，测到关键链f后才算S_load。不可用1.65×或1.19×直接推总加载节省。

执行边界不变：不触910384/n17/n18，不读nonce/rawargs，不重播未知GPU执行，不引入GPU F64。落实记录只需现有卡的采纳/拒绝及实际句柄、准备/排队/加载/测量时间，无新框架或报告门禁。
