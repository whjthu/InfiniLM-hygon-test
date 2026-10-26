Astra 113 已完成。输入及12项来源SHA全部匹配；仅消费 Deep64 family 与 GLM749 两轮，提前触发原因是重复可避免工程错误。Kimi/GLM在途Model继续，不等审核。

- **Kimi优先原33卡Model，追加同驻留C2值得做。** 旧C2用户中位14.6103 TPS距15只需2.667%速率提升，但必须报告每个用户达标情况及全部33卡goodput。固定前8题、16个ABBA case预计额外5–10分钟；不为C2再付约31分钟冷load。原38优先、旧39文件SHA不变，runtime/stop/硬限不改。root增量已发布37a8059并发DATA APPEND GO；尚无已执行或性能证据。当前源码支持动态queue与未执行原因ledger，不能把“冻结”变成额外冷load规则。
- **Kimi高并发主要困难在target范围。** C8W8/C16W4同flat64，周期1.045/1.061秒，target host占88–90%；接受率70%/87%，不是并发越高接受越差。删除全部draft代价的条件收益仅约5–7%。即使每周期全部接受，要15TPS仍需C8周期缩短49%、C16缩短75%。这些是固定周期假设下的阶段数值，不是硬件roof；GPU计算/通信/CPU/同步各份额未知。
- **Kimi第二卡仅为一次INT8 whole-MoE组件。** 751终端已COMPLETED0，实际6条W2+1条layout I8指令，尚无GPU数值/性能。补静态expectedheader/专用keeper只是工程准备；仍无GPU GO。原计划T8/T64/constructed lowreuse一次对照，先256整数atom oracle，再whole-MoE四配对；不直接加载整Model，不按峰值估收益。
- **GLM优先继续755原16卡Model。** 749组件T16/T8速率中位+10.22%/+9.33%，各4/4配对正；不是Model收益。755正双stage加载，c650仅T8/T16命中，原32arrival OFFONONOFF验收自然EOS/QA/用户速度及每卡吞吐，其他T混合收益未知。
- **Deep C4保留opt-in，不泛化。** 两臂62/64正确、64 EOS、0 cap/解析失败、无正确性discordance；64用户decode全部变快，但未证明质量非劣。n05任务wall235.337→242.684秒（+3.12%），goodput−3.03%。其中584题113→196 token，decode时间40.107→60.330秒，说明更快token不保证更快完成任务。保留此题、长度和tail，不能挑题美化。每node吞吐除8卡，不把4个异时replica TPS相加当集群实测。所有>15TPS正确EOS goodput仍为0。
- **Deep MTP暂缓近期GPU卡。** 当前CAPI没有所需hidden/all-position输出和model-wide rollback/prefix commit。K1即使同regime乐观2倍也仅约5.94TPS，距15仍远；是新native功能，非配置开关，首可信速度ETA未知。

112落实：K112A实际加载；D112A完整终结并本轮消费；G112A组件通过并已进Model；L112A绑定/随load计数已落实，但map/H2D分阶段时间未知。不能把INFLIGHT或字段存在记为性能成功。

最小工程修复：复用成功父对象、头文件和完整LD环境；已有封包统一UTF8无BOM/LF；receipt最终落盘后再绑定wrapper SHA；固定实际CUDA_ARCH750/gfx936通路，禁止复用导致空body的HIP宏。已知机械修复一次有界继续授权，不逐条等待，不加新框架。INT8五次失败可见CPU step共62秒，端到端准备等待未知；避免再发生远比额外报告有用。

JSON给出逐卡owner/资源、采纳条件、样本、退出、替补及ETA。C2剩余时间不足沿原deadline跳过；16×180秒保守预算可达48分钟，5–10分钟仅估计。无GPU/SSH/Slurm/Git/新agent操作。按当前写权限将交付存于可写根目录；来源远端嵌套证据未重新获取，限制已标注。

冻结增量：C2 staging因Windows反斜杠key在首link前失败，owner报告原39/live新队列均0变更；root已同scope授权as_posix规范并只补缺项，尚未实际admit。INT8最终source准备完成、push在途，仍未GPU。以上依据父任务消息，不冒充远端亲验。