Astra review 118 — 消费三轮结果，转向真实验证路径与显存失败

结论：GLM M15 保留 default OFF，停止八 offset/M15 微矩阵重复投入；GLM 下一项只落实已有 G115A 稀疏全模型归因。Kimi 首先闭合 478 的真实显存失败，消费已经留下的 C2 诊断，不重跑18-case矩阵；验证路径是主要可优化区间，控制/遥测不是优先对象。TP32 full93 继续拒绝加载。Deep 继续既有前缀资格工程，不列为性能收益。

本次只本地读取源码/证据并写报告；未 SSH、Git、上传、编译、Slurm 或 GPU。消费 910376.761、910375.764、910376.764 三轮。117 为预加载源码咨询、零轮。审查期间到达的 478 失败作为附加负诊断消费，新增 qualified Model wave 为0；不可为了凑轮次把12个case拆成12轮。

实际结果与上一轮落实

• GLM761 八逻辑 offset 用训练层权重验证4.4716–4.4976x组件倍率，实际物理1卡、构造M15输入；只证明组件及清理复用。
• GLM764 16卡TP8/PP2，同32问题ABBA、每arm两epoch。task 13.203829→13.326042 tok/s（+0.925588%），decode visible/CAPI 22.723909→22.914066（+0.836813%），user mean1.164790→1.177654。两arm每epoch31正确/32自然EOS/0cap；0人>15。总wall335.508733→332.882029秒（-0.782902%）。仅两个重复epoch，不能建立稳定普遍收益或正式质量不劣结论。保持default OFF，不把component4.5x转述为Model大成功。
• GLM emitter stage1 A0中T16有91call/45.911945秒/median503.571948ms，占该epoch decode95.869156秒的47.89%；T15仅10call/8.572658秒，ON为7.625160秒、median761.857349ms。T15投影确实改善，但更大T16与prefill未解决。两PP stage时间重叠/嵌套，不相加。约140–142秒 pooled prefill也不是M15候选的因果收益。
• Kimi764 TP32一层MoE T8/T16/T64分别1.418964/2.077841/5.751677ms。8 distinct历史行，T16重复、T64再重复；相对TP16旧~1.3002ms跨轮/不同provider不能当配对。没有多卡必快、93层收益或full93 fit证据。
• K116A的六HOST实际编译已完成；K116B真实C2观测已产出，但整run478 FAILED1。最新r9339确认下一MMLU prefill后4GiB allowner reserve断言；四节点同报，native_error与cleanup_error均null，postKFD通过。health.error_present=false不能覆盖target FAILED和emitter_error=true。不得猜成质量screen或降低floor；不得盲重启。
• 478留下2warm+8C2 profile+首8MMLU两arm共12case。首8MMLU每arm6正确/7EOS/1cap，screen_gate.stop=false；剩24题没有结果，完整MMLU32仍未完成。其吞吐只是失败运行内局部样本，不能晋级。
• Deep452闭合原13对象/链接与9HOST契约；451的两fmt弱符号审计失败保留。r129本地9HOST事务测试已通过，但新增inspection/ticket/fault DSO仍未真实编译，更未GPU执行。不能把local HOST proof当43层原生状态正确性。

Kimi C2新诊断必须立即消费

四组observer ON/OFF aggregate变化依次约-0.198%、-0.110%、+0.337%、+0.062%，输出计数与8题正确/EOS匹配。它们仅8 distinct GSM题、旧套件子集，不是质量扩展。真实T16记录24/23/16/14个cycle，总77；记录cycle总103，不能将全部区间标T16。

在同一个emitter时钟内，verify CAPI / recorded cycle依次87.59%、89.22%、89.32%、89.35%。state replay whole约3.1–4.9%；guard约0.56–0.62%；feature harvest约0.3–0.44%；progress约0.5%。这证明先优化verify调用内工作比删guard、改日志或微调Python更合理。CAPI包括PP/依赖等待，不能称纯GPU计算或纯MoE。node0 draft RPC约0.38–0.71秒；它和emitter等待不能跨主机相加。stage0 selection-control大等待是末级结果依赖，不是已证实可删通信。

发现并已告知root的确定报告缺陷：phase_observer_r126.py 的prefill_interference在observer OFF没有micro timestamps时返回0.0，实际上应为UNKNOWN/null。ON组第一用户可见区间内第二请求prefill约1.996–2.197秒；OFF的零不能解释为没有prefill干扰，也不能当observer新增成本。最小修复仅将缺测置null并声明coverage，或从已有同钟事件补算；不需重跑GPU。first token时间不可后移以美化decode。

源码核对与方向选择

当前GLM glm_arrival_probe_r129.py仍只取modern_glm_pipeline_last_profile()六值中的前3个。实际d940...文本model有6值、每次forward创建/销毁4个event，固定B8窗口不能覆盖自然T16/T15/prefill；incoming recv/allgather与layer依赖重叠。优先复用G115A已有源码绑定和稀疏observer方案，而非又写一个未绑定profiler。该路径为TP8 PP2、定制projected/DSA，不符合上游paged graph的PP1 ordinary unquantized KV条件；不能直接部署该分支来提速。

Kimi batch_cohort确实逐请求prefill、W8边界、真实packed verify；batch_native.finish先selection control后feature harvest，state replay另有collect；这些阶段已被当前观测覆盖。当前数据不支持先删控制屏障。micro2早在105/106讨论过，不能再次当新想法；更不能现在用扩大prefill批次来试探已失败的显存。W16不是改一行上限。

上游main仅比当前5f91ad后续3commit，Core/RT/Ops/CCL同pin；adopt-modern为分叉。W4A8源码确有shuffle/fusedMoE，但packed I4/per-output F32与现有FP4 group32不同，localI%64：TP16I192合、TP32I96不合，Hygon provider及转换误差未证。保留为CPU/source替补，不新开完整模型路线。DeepEPDispatcher四入口仍throw stub。TP32 full93静态58.906441GiB且扣4GiB仅余1.093559GiB，driver/CCL/峰值未知，禁止full93构造/加载。

下一步只执行这些卡片（root登记采纳/拒绝/暂缓；沿现有授权执行，不新增用户索权）

K118A — 第一优先，Kimi ctx owner，零新增GPU，20分钟首判。
假设：失败由下一个长prefill的实际临时或保留内存越过floor，而不是质量screen。读取已有failed_memory_guard的label、每owner minimum/latest、case-start/reset、上一case结束、prefill长度及T；将allocator/native持久缓存、taps、logits与Torch保留分开。最小源码修改为缺测prefill overlap=null及guard失败摘要包含32owner minimum/latest/phase/shape；成功路径不减检查。若能定位一个确定保留对象，修其reset释放/复用，复用同父对象和编译上下文；不能凭torch reserved直接对native allocator调用empty_cache。只在实际容量修复闭合后用导致失败的原case先做短资格，再接尚未完成的24题OFF/ON，保留原首8结果的不同run provenance，不重跑C2或18case整矩阵。后续如需Model使用原32target+1draft、4GiB floor、原问题和cap1024；加载约历史28.5分钟，排队UNKNOWN，测量上限20分钟，失败立即保存并正常收尾。若无法解释峰值或修复保证，停新Model，不降低floor、不删长题、不把未完成当通过。

G118A — GLM owner，落实旧G115A而非新微矩阵，CPU闭合目标20分钟，之后最多一次16卡TP8/PP2加载。
假设：T16/prefill大区间内存在值得改的投影、attention、FFN或依赖等待；目前具体可移除量UNKNOWN。最小改动：当前b588/1760 native父对象绑定；当前bridge owner线程接入sequence+完整(row,past,count) selector；读取6个pipeline值；原stream预分配有界events（<=512/owner），coarse逐层+PP边界，detail仅每stage一个真实full/shared-indexer层；不增加sync、不改math/schedule。先闭合private8实际源码与bridge hash，不能替换成旧8812。一次相同32arrival episode controls-before / sparse-profile / controls-after，选自然T16 3calls、T15 3calls、B8 PREFIX及B1 PREFIX；不存在匹配则UNKNOWN，不造padding。测量<=15分钟，加载/排队独立记录。source/owner不匹配、扰动掩盖区间、质量/清理异常立即停止。只对一个最大、至少占实际critical wall约10%的可改区间出下一kernel/调度patch；若没有就停本路线。本卡本身不宣称加速。

K118B — Kimi verify owner，容量闭合后的唯一性能归因/实现卡，不能与K118A重复加载。
由已测87.6–89.4% verify区间出发，在同一修复后质量加载中只采真实C2/W8、当前T16、当前W2pair路径的稀疏事件。沿当前e955六HOST/math对象接入owner原stream、预分配events；覆盖每stage的MLA/KDA、routedMoE/sharedMLP、TP/PP边界；两warm后四个代表性verify窗口，ON/OFF包络控制，最多10分钟附加测量。不要开启已知逐段sync的PROFILE_R9125，不用739 T8/T64或层1历史fixture冒充当前层。若MoE占最大预算，复用已存在FP4 W13/W2pair实现，先仅改实测最慢的一个phase（T16实际router/job形状与回退保持）；若attention/collective更大则只改其具体边界。没有>=约8.7%逐用户首末时间潜在可移除区间就不承诺跨15；相同token预算的8.7%是目标设计量，不是收益预测。先四对真实cohort证明收益超过自身配对漂移，再扩大既有GSM32/质量benchmark；不得进行形状/精度盲网格。若run因容量未恢复，本卡CPU source准备并PARK，不新增GPU。

D118A — Deep原owner，沿现有r129继续，不另立加速试验。
先实际CPU构建新增13HOST/adapter、ticket和fault入口与已知fmt例外校核（目标30分钟）。通过后沿现有8卡TP8PP1资源单次加载16case：2/3/126/127/128边界、mixed accepted1/2、全43层长度/validity、cancel/stale/partial-publish fail-stop/reset及继续。测量<=20分钟、4GiB floor；原计划2400s硬限、2300s内限。任意owner不一致、未来字节可读、错误复用、容量失败即停。EOS/cap为harness模拟，没有MTP head，没有自然文本quality/tok/s收益。主性能资源仍优先Kimi容量恢复和GLM大头归因。

明确停项：M15八offset/窄矩阵重测、TP32层数外推/full93加载、Kimi INT8v1/v2盲调、GLM664 selectiveINT8重复、已测724 prefillpipeline重新命名、未经生命周期证明的power2 padding、空DeepEP、依赖整栈换代。当前无HTTP服务或全用户>15结论。

成本：764 GLM实际26m57s；478 target41m46s、draft40m25s、keeper42m21s失败，已留下12case。当前证据不能拆出全部准备/排队/加载时间，不伪造。主教训是复用这些结果并直接定位容量，而不是再付一次全模型加载来重做已经完成的C2归因。
