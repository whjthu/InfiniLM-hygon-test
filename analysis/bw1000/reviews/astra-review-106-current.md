# Astra 106：立即恢复实际判定

输入 SHA256：0b7122f53ec6f64c6a5792250cd015991c22d389e595acdb89cf9a2ddf12a8dc。仅消费 **910376.714** 一轮；新增整模型收益为零。输入的10份证据和GLM新增producer及7个子文件哈希一致。

105落实至106输入约6小时15分未出现新整模型轮次。CPU通过、两次affinity失败及Git心跳均不能算性能进展。GLM714仅组件：M17/31 wall下降14.79%/9.25%，M32反退3.11%；保留M16，结束lease网格。

|顺序|决策|资源与首次判定|退出及替补|
|---|---|---|---|
|1|Kimi B已GO继续，跟随唯一invoke，不重复提交。C16 AR/fixed/adaptive先，随后C32。|65卡；预计总40–65分钟，复用已通过12节点CPU/native b4。|无超噪声收益保留最佳AR/fixed；不再缓存网格。|
|1|Deep先纠正实际dispatch并准备当前decode profile；小T只做实际入口T2/4/7尾段组件，B1及T8/16/32 fused零命中对照。|1卡；已有CPU395及47对象，预计10–20分钟。|叶子收益不代表B32；8卡必要诊断复用现有profile，预计25–45分钟，先量当前瓶颈与<8尾段wall份额。|
|1|GLM只修最小接线：旧M16/M32队列改成C32M16 serial/deferred ABBA，严格逐cohort开关。随后条件16卡同载短warm和配对。|16卡；接线10–20分钟；加载时间未知，启动后首判规划25–60分钟。|WORLD alias调用顺序匹配但真实CCL未测；短warm超时停所属run，不盲重播。无收益保留serialM16。|
|2|**明确允许独立A**：B有实际启动/推进后解park，不等B全部结束。C16 prefillmicro2短容量warm，然后C1和C2/4先做8题配对，再扩32题。|33卡；预计35–60分钟首判。原4GiB guard不动；共享I/O有干扰则错开权重读取。|容量失败保存快照并退出，不降floor。不要等127-case队列才给结论。|

Kimi最多B、A两张实验卡；旧16卡MoE是替补，不是第三张卡。已有两次失败的修复点是first execution::submit之后验证owner4，不能手工提前绑4或在Runtimecreate后立刻验证。复用c390和原头，不再拼环境。

GLM源码支持继续做实际warm：forward使用WORLD send/recv，TP collective留在TP，反馈使用WORLD allreduce；两边send0/send1/reduce0/reduce1与recv0/recv1/reduce0/reduce1匹配。同步CAPI不排除不同stage进程重叠。CPU八组EOS/past/双logits通过并不能证明CCL进展或实际收益，也无需再造独立CPU模型或强制新communicator。理想两micro均衡上限wall约减25%只是条件上限，真实收益UNKNOWN；同时看发射聚集造成的ITL。

B65+A33+GLM16+Deep1可形成115卡有意义并行；Deep转8卡Model后122卡。其余空闲卡不应做重复INT8/CCL负路线或空占。优先确保实际句柄、一个加载内完成相关对照、结果到达即消费。

root报告长期goal仍blocked，而用户已明确继续。当前授权工作应继续；通过现有人工Resume/Continue入口恢复goal自动续轮，不能声称代理已自行resume，也不临时建timer。Git心跳结束且所有ownercompleted时必须如实标记停滞；下一授权续轮立即恢复原owner，不能写“独立继续”掩盖无人执行。

质量按任务答案、自然EOS、截断、状态正确性和后续真实benchmark；不以token/中间值bitwise准入。8/32题均为快速筛查。完整参数、样本、来源、估时、退出和替补见JSON；root逐条记录采纳并执行。


**Deep关键更正（最终结论优先）**：r9222漏看r8936外层dispatch，普通8<=T<=64且几何合法时已走fused validate/compute；precision5仅改变attention collective为F32，sink计数0不能推断逐query scalar。保留原错误证据并追加更正。CPU395候选只可能影响实际T2/4和EOS<8尾段，T8/16/32是fused零命中对照。当前B32 profile/dispatch因果判定优先于叶子优化；一次必要8卡加载可同时取普通decode和自然尾段证据，正组件也不能自动晋级为B32收益。详见JSON D106A/D106B。
