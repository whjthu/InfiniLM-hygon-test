# Astra 正式性能审核 102

输入SHA：`18aa8ea00cd9b8671091d45dff2b1b605f91cff9f51fe196d11f4aeabfda303c`。12/12冻结来源已验SHA。仅写本报告与JSON；未改ledger、source、live或提交GPU。

本轮消费 **416、697、727+487+541（合为一轮）、554** 各一次：三组件矩阵＋一Model矩阵，共四轮。它们未被101消费；部分GPU较早结束不等于已取得独立资格或已审核。545/550、CPU/probes计零；Deep1065不重计。root实施102时原子更新ledger并归零待审核计数。

## 审核结论

- **Kimi head416：支持进入已准备的shard-only Model。** M32 full/shard host p50为2.154/0.692ms，paired mean约2.926x；M1约6.099x、M64约1.844x。独立进程constructor每owner实测节省2.05078125GiB，M64 held节省1.98828125GiB；仅组件成立，不能认证Model容量或乘到整机加速。新ModelCPU366通过仍计工程零，4GiB门不变。
- **GLM697：组件已过，701正在真实一次冷加载。** B32完整helper 1.940→1.232ms，四对皆正；synthetic uniqueKV不算任务质量。Slots/caller/scheduler三HOST闭包已做，不再重复组件或边界修改；让701同handle完成。
- **Kimi真实PP4：C32 common30.410→33.012 tok/s（+8.56%），65卡0.46785→0.50788/card；用户中位1.44486→1.44916，仅约+0.30%。** 输出token6658→6896，wall218.94→208.89s，故不是等工作量8.56%延迟缩短。C16两组按总token/总wall汇总16.283→30.212（+85.54%），state用户中位2.747/2.812。两配置均32/32正确自然EOS、cap0，但仍远低于真实用户15+。这是包括strategy/I192/draft/frontend的运行比较，不是单GPU算子因果收益。
- **Kmax554：23case组件过，目标接受率未验证。** C32K1完整query217.035→206.296ms：**延迟下降4.94775%，query-rate提高5.20529%**，两种百分比不能混用。profile中head约91.18ms、context-proj40.48ms、五层71.05ms；3D非contiguous输入是可测布局假设，尚非已证实原因。K7与maxKs=7混合组近乎无收益符合预期。

review101的Kimi矩阵/两个组件、GLM组件与B32闭包均已实施；Deep仍须实际对象链确认后做内层归因。旧controllerfalse、545/550启动失败与后续独立资格应一起保留。

## 关键路径与下一步（每模型最多两卡）

**K102A：32卡shard-only Model＋轻progress实测。** 使用已准备CPU366/DSOb4f916的冻结manifest，910380 n08/n09/n10/n12，32DCU、每节点c64/128000MiB；所有owner实际create/load/finish/warm/max-context>4GiB后同ctx继续，draft另占空闲1卡。先冻结有界progress与分段计时：serialize、write/replace、control收发/屏障、guard/readback、nativeCAPI及有界GPU event。最终完整报告移到请求计时结束后，在线必要安全/control仍计入；保持真实first/last/TTFT，不减掉日志时间伪造用户体验。

当前每8step全历史save的源估算是单emitter至少7,109,054,967 bytes、334 callbacks，**不是实际I/O字节或耗时证据**。C16 common与CAPI之间的大gap需量化各不重叠区间及残差；不能全部叫I/O。用一个短、同Model、可比历史payload的legacy/light control窗口判定保存代价，避免为日志重跑64卡。AR/state使用同一轻progress策略；新PP2对旧PP4的结果只报组合配置Pareto，不声称纯head加速。容量失败即保留证据停，不降门。

**K102B：已授权n15单卡2D head实验。** 910371/a02r3n15/1DCU、c8、64GiB、15min；与PP2 draft使用串行。对比Kmax3D与显式contiguous→[B*K,H]→原head→view[B,K,V]，计copy/alloc/head/selector/D2H及完整query；保留K1/K3/K7、mixedKs和C1/C16/C32必要控制。仍保8hidden、attention/context数学；不启用CTX cache。合理数值/shape/state通过且完整query改善才进入一次真实target接受/吞吐测试，否则停，不能承诺7x。

**G102A：只完成701。** 现910376.701 n12/n13共16卡、每节点c64/240000MiB、manifest19e3023，fresh32同驻留M16/M32。收terminal/audit/post与已有诊断，不重启、不热改、不重载。以正确自然EOS throughput/card、用户rate、TTFT/ITL共同选配置；aggregate升而用户降时保留明确Pareto点，不宣称普适更快。

**G102B：条件式一次wait/control归因。** 优先消费701已冻结的profile；没有时仅在下一次有必要的同驻留机会用已验证hook取短窗。不可为了profile再冷加载，也不可把stage1 elapsed两倍叫计算两倍。查明一个关键等待后才选一个调度/control改动。

**D102A：立即闭合实际SWA/CSA/HCA hooks。** 确认EDF `.d`/object/symbol及AttentionSectionsR756 envtag、eventfinish/drain、journal到底是否链接执行。复用兼容部分；缺失仅在当前attention TU加最小marker。910374/n05/8卡、c64，真实B4/1681token短OFF/ON/OFF、即时read/reset；journal/drain单列开销，不把每调用强制同步带进uninstrumented成绩。不再粗测1065，不建general profiler。

**D102B：归因后唯一改动。** 同8卡资源，只对D102A明确的主导projection/cache/core/collective/host-wait代价动一个点；完整attention配对后同负载Model验证，保B32 compaction及B8/B16行为。没有可靠主导项或完整收益就停，不转HC/MoE网格。

机械import/path/encoding/accounting启动修复可在同资源、数值不变、前任务terminal/postempty后，冻结immutable successor并经短CPU检查直接继任，避免反复人工GO等待；数值/资源/新Model仍冻结准确source/claim。轻progress应可观察、保留最终证据，但不能把省日志称算子加速。允许不同tokens，先合理、后benchmark无明显掉点；禁旧bitidentity/F64门。没有新physicalBF16peak/GDR证据，不猜硬件上限。保护910384及所有live，独立已授权实验不等待review。
