# Astra 125：结束调度支线，推进有新机制的计算候选

结论：这3轮没有新增整模型加速。Deep159和GLM chunk128均停止默认推广，不再试插入频率或chunk中间值；W13 wavepartial保持PARK/OFF。下一GPU优先CPU512已编译的FP4直接解码组件r161a；独立资源同时推进一次必要的Kimi r162加载。TP32/I96 r163值得保留为计算卡替补，但不抢r162的同组32卡，不把58.9GiB静态估计当整模能放下。

本次仅只读本地源与结果，未SSH/Git/Slurm/启动GPU。新文件125不覆盖124。主要证据：model-terminals-root-r9481-a.json、CPU-candidates-read-root-r9476-a.json、r161a源码/CPU512回执、r162 MODEL-REVIEW及CPU合同、r163源码与几何合同。

## 124落实与实际结果

| 实验 | 结果 | 决策 |
|---|---|---|
| W13 1171部分L1+1175独立L47 | 计一轮；12配对W13 +9.95%～11.91%，full-MoE +4.22%～7.92%，原每block≥5%条件未通过 | 组件小幅有效；专用整模PARK/OFF已落实，不重复组件 |
| Deep159 /1178 | 27m03、正常退出/post1179空。A/B各128执行、64独立题×2，均124正确/128EOS/0cap | 调度B停止推广，保留原A |
| GLM160 /784 | 27m51、audit qualified、16owner/IBext合格、post785空。full/chunk/chunk/full同32题各31正确/32EOS/0cap | 保留deferON+full；chunk128停止默认推广 |

Deep task 19.425→18.753 tok/s（−3.46%），common decode 36.227→35.383，逐用户p50 1.1736→1.1478（−2.19%），TTFT p50 35.32→38.27s（+8.34%）。首二token gap p50确为28.233→0.640s，不能以局部gap改善替代用户速度；早decode成本必须留在原时钟内。

GLM chunk task端点平均15.856→15.151（−4.44%），decode CAPI 25.548→24.104（−5.65%）。PREFIX 4→7调用，decode 209→251调用，PREFIX约50.5→52.3s、decode约85.5→92.7s；这直接解释调度策略增加了实际工作。ITL max约39→13s有局部价值，但用户p50仅约0.863→0.868、TTFT约26.9→29.7s，不能作为通用收益。

GLM新增C1-first8为8题串行接纳，8/8正确/EOS：用户p50 5.0086、decode CAPI 5.357、task4.081。C8同first8为8/8正确/EOS：用户2.6398、decode CAPI16.391、task11.581。C1 TTFT p50 46.25s包含同批到达后的排队，不是孤立单请求TTFT；其稳态ITL约0.199s则说明低并发仍有真正的计算/通信延迟，单调度不能把它变成15tok/s。

本轮已有两项约27分钟实际GPU实验并正常收口，纠正了上轮只消费心跳的停顿。不能因结果为负再重复三轮。CPU511 registry/clock、CPU512解码编译属于准备，不计GPU性能轮。基础质量未见掉点，但重复同题不冒充更多独立题；后续真实benchmark仍不要求token逐字一致。

## 下一轮卡与顺序

**K125A：首先r161a directFP4组件；owner为Kimi原执行者，n01+n04/TP16/16卡。** 假设W13的FP4转换指令成本可减少；不同于已PARK的归约，保持原FP32归约、LDS、W2pairON，只改正常值直接解码并保留边界fallback。复用CPU512 Native57c6bed3…、已过65536组合CPU合同及原对象；先读已保留的确切两kernel ISA，不能拿源码ldexp次数当实际动态指令。若编译器已生成等价代码且没有可辨识新机制则PARK，无需GPU；否则沿用4warm和3组OFF/ON/OFF，两输入、L1/L47各独立fresh进程，避免已知跨层RSS问题。首层预计2–5min、两层预计5–10min，队列UNKNOWN；这比支付整模加载先获得一个收益/否定结果划算。沿用现有finite/NRMSE/tail及full-MoE/W13≥5%、OFF漂移≤3%条件，不另加门。负结果停直接解码，不再改相邻常数重测；组件成功仍不自动整模晋级。

K125A的替补为r163 actual TP32 I96：新TT2Tile4/I96+W2pair对同TP32 generic，T8/T16/T64/受控Gaussian64、现有两组ABBA。先完成其尚未执行的实际HIP构建；这是新候选，不重做旧764 generic或767 constructor。仅在r162完成/释放同组32卡后做一层组件；预计构建数分钟、GPU约5–10min，均为工程估计。沿用现有≥10%同TP32 full-MoE设计筛选，不把它当整模速度或容量门已通过。T8可说明小verify形状，不冒充真实C1服务。若无净收益停止I96模板路线；若有收益，下一步先按实际峰值补容量判断，fullModel仍UNKNOWN_STOP，禁止直接加载赌58.9GiB能放下。

**K125B：一次必要r162加载；owner为Kimi原执行者/root，原32target+1draft资源。** 采纳现成cacheON、W13OFF、Native798c4bca…、原445/136b协议，完成已有CPU import/preflight后执行；与K125A仅在物理资源确实独立时并行，不等待审核、Git轮询或r163准备。历史加载25–30min，C1四题+C8两组共20请求/16独立题预计再5–15min；首个flush归因可较早出现，完整判定约35–50min，队列UNKNOWN。完成即退出，不加临时矩阵或延长驻留。

r162的可证伪问题是“cache生命周期在真实Model成立，并能量出当前Native剩余时间主要在哪一段”，不是“单次证明旧1180–1408s由state_dict造成”。固定cacheON没有同次OFF反事实；两次state_dict构建和flush计数只能证明机制执行，历史冷暖/IO不同不能得加载加速倍数。读每owner state_dict/cache_lookup/load_bindings时间、每flush进程CPU/fault/实读IO及wall；不把8owner CPU时间相加当关键链wall，嵌套inclusive桶不相加，H2D DMA仍UNKNOWN。若当前registry份额很小而Native仍大，停止继续优化registry，转实际剩余主桶；若加载明显较快，也仅称当前观测，不宣称因果倍数。pilot报告C1/C8真实用户和总吞吐、正确/EOS/cap；“20/20”是现pilot字段条件，不应因单题错误触发强行逐字对齐或无限重载。

**D125A：保留production ac937、原B32吞吐路径，停止159；owner为Deep原执行者。** 下一步先用现有1178窗口/源码做一个具体decode成本定位，列明准备修改的算子、当前真实形状和可复用组件入口；不再重载验证batch/早decode。首次离线判定预计10–20min，GPU时间为0。只有出现真实新机制后才开同形状组件配对；当前未见这样的源码，实际GPU收益与时间UNKNOWN。C1补测继续只搭下一次有具体候选的必要加载；124中这项尚未落实，本轮不再为遗漏单独付加载。无需填第二张无差别实验卡。

**G125A：保留781 defer收益及160 full基线，停止chunk策略扫描；owner为GLM原执行者。** 以新C1稳态约0.199s为目标，先复用已有源码/trace核对该形状是否已有可行动的operator或collective主桶。若证据缺失，只准备一个同C1/C8形状的短profile，搭下一次有候选的必要加载；不重复767无actionable桶的宽扫描，也不再为了从5.01提升到15试batch2/4或chunk64。离线准备10–20min；此刻没有合格新kernel候选，独立GPU重载暂缓，收益UNKNOWN。无需凑第二张卡。

## 有效边界

Kimi scalar FP32处理FP4，333TF BF16 dense不是该kernel物理峰值。r161a源码成本账中的逻辑读数/ldexp调用/不同专家存储量均不是实际HBM流量或动态周期，ISA静态计数也不是动态占比。沿用S=1/(1−f+f/s)：假设真实关键链W13占0.33，区间1.10×仅对应整链1.031×，1.20×对应1.058×；当前f仍是稀疏层条件外推，不直接乘用户p50。r163改变TP/PP及通信范围，既不能据I减半宣称2×，也不能用同TP32组件收益替代TP16PP2整链对照。

GLM从C1用户约5.01到15需要约3×端到端改善；在相同时钟下即使把某子项完全消掉，它也必须占至少约66.6%的时间。没有这样的实测主桶时，不以硬件峰值承诺目标。Deep/GLM均不能用重排相同计算消掉稳态算子成本。

已有边界不变：不碰910384/n17/n18、不读nonce/rawargs、未知GPU状态不重播、无GPU F64。r161分析器单次仅一层两case，字段all_four_controlled_cases_promotable命名不能当成四case已通过；root沿现有记录跨L1/L47汇总即可，不新建框架。落实只需更新采纳/停止、实际句柄和准备/排队/加载/测量耗时。
