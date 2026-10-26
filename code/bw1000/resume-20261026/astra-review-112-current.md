第112轮审核完成：只消费375.739、375.743、376.743三轮；固定输入及11份sources SHA全部通过，另验GLM停止增量3份SHA。此次没有新增整模型或真实服务收益证据，Deep四replica在途不计完成轮，bitdecode源/历史停止不算新性能轮。

- Kimi：W2全MoE链T8/T64配对速率分别+5.14%/+11.70%，两点均4/4正，推进已授权的一次33卡严格W8 OFF/ON Model。source/CPU仍在准备，尚未冻结发布或启动GPU Model。计划C1 screen8→C16 GSM64→C32同时到达但native cap16排队→custom numeric-choice MMLU32复用一次加载。主看正确自然完成任务wall、质量goodput及全部33卡吞吐，并列单用户>15tok/s、TTFT/ITL与队列；保留首8早停，不新增门槛。
- Deep：1118/1119/1124/1125沿原64题与每node首8早停继续，不再启动、不扩样本；汇总独立质量和node内配对任务wall。
- GLM：bitdecode已STOP，无编译/GPU。已保存ISA与原receipt对应，有v_ldexp与NaN常量选择；旧r1458八owner均负约-3.94%，旧scalar/current vector load差异明确，不能称同binary。唯一替补CPU候选是当前down同route相邻两output/CTA共享activation load/decode，权重仍两份、两路原W32V4/FP32归约独立；先历史去重/实际绑定，相同负例即停。条件GPU仍须原source发布GO，不新Model，不从Kimi收益外推GLM。
- Loader：实际r2515已经bulk且scalar0；五counter为whole-session，非load-only或H2D总字节。CPU先区分每flush state_dict映射和completed copy，设备边界读数搭载Kimi合法load；不得让仪表延迟Model。复用r9212d缓存，不重写旧逐字节gather。

优先级为Kimi一次Model、Deep原在途完成、GLM短CPU判定、Loader不阻塞诊断。每卡owner、资源、首结果ETA、必要样本、退出和替补见同名JSON。GLM95.95%仅定位局部链kernel活动，强扰动trace比例不代入整模型Amdahl。允许不同token，无F64；保持指定固定上游基线，不静默切换另一个modern分支。
