# Astra 第111轮审核

本轮输入及13份sources、截止后Kimi两份补充SHA256全部匹配，只消费 Kimi466/580、GLM737、Deep1112 三个qualified矩阵。

- **优先Deep C4。** 1112原audit0、post1113空、canonical02c03f5e/producer9b7531a7闭合。C4 decode +17.15%、任务wall -14.25%，两臂8/8正确EOS；每用户median 2.276→2.590 tok/s，仍远未到15。保留默认OFF与受控C4 opt-in，下一轮采用4个8卡replica分摊64个冻结独立题：parent374 n01/n04先行、n05/n07错峰，须fresh确认健康自有空闲；每replica先8题再完成16题，节点内配对、只算一wave。预计首块20–30分钟、64题30–45分钟（排队/共享存储拖慢另计）；若前两节点load恶化则不扩后两节点，退回双replica32题。不重复C1/C2/C32网格：C1/C2的953正确性方向相反，C32虽decode +2.20%却任务wall仅-0.193%。原8题跨C重复、C32又包含它们，不能合池当独立质量样本。
- **GLM暂缓G110B独立16卡load。** G110A已落实，737 T13/T15组件+5.5%且各4/4正；但whole730只有12/196 calls、10.14s/170.98s HOST暴露。粗略同份额估算也只是约0.3%wall量级，且不是预测/上界。先用现有单rank真实权重链固定T16（HOST48.29s）分段定位，最多T2一个替补；20–40分钟得首结果。确认一个主导可改操作后再单改动，不盲Tgrid。739仅CPU Model DSO就绪，已有组件门不自动触发昂贵Model加载。
- **Kimi选择W2相邻双输出CTA候选。** epsilon.10不采用：rawTPS +3.31%伴随tokens +3.83%、wall +0.50%，正确EOS goodput下降。截止后739/425终态、audit及post740双空已闭合，两份新增SHA验证，留nextwave1，不增加本轮consume。trace缺失只在每ON窗rep7，所有owner统一剔除rep7后每T224内部样本完整。T64 W2约2.724ms、W13约2.607ms；先只将W2两相邻输出共享activated输入，保留各自FP32累加与原route归并。CTA减半不推2倍性能，同fixture T8/T64无profile整链ABBA决定。profile开销约36%/7%，不从区段和推Model/PP瓶颈，不为报告重载33卡。draft质量、典型接受率、INT8/INT4仍是原则性替补，不能凭空开新路线。
- **Loader转native_load1399s拆因。** 426真实2.18GiB shard组件通过，但每arm是两个并发reader；warm mmap/memfd同PSS1.170608GB/reader，没有共享优势。memfd copy .818s、总ready11.117s不能解释31分钟加载。优先15–25分钟完成当前成功loader源绑定，再用单层真实栈分TP切片/packing/CPUcopy/pagefault；H2D/同步复用下次正当load。不给报告另造coldload。423/424/425机械失败仍0wave，safe_open/native Model兼容未证明。

110的Deep与GLM组件卡已落实，Kimi当前源绑定/组件归因也已执行；本轮改变的是低暴露GLM的加载优先级。新增终态证据下Kimi739也已结束，Deep/GLM已终态、Loader为CPU；旧status空表与reservation不能代替新句柄观测。

精确BW1000物理峰仍UNKNOWN，旧dense BF16/MMAC不外推FP4 MoE。允许不同tokens，质量看真实benchmark无明显掉点。详细owner、资源、退出/替补与ETA见JSON；root需登记各卡决定并派发，独立试验不停等审核。


截止后Loader源结论已纠正：working copy.cc过时，实际04a6a1a8 DSO已调用copy_cpu_strided_r2515并更新copy_calls_r2515，不能说当前逐字节memmove未优化。root已取消suffix候选。Loader卡改为先绑定实际helper/source/统计接口、读取现有r2515 counters，再做单层真实路径分段；tp_dim1 drain和H2D(false)同步仍需测份额，1399s收益未知。不新coldload、不重复实现已有优化或r9212d缓存。
