第107轮结论：继续驻留中的Kimi A/B和GLM ABBA，Deep优先做一个保留indexer/state语义的CSA批量回读实验。

- 本轮消费1086、3740、1090三轮诊断/组件证据。1090独立主机路径复核已通过，原audit1保留；没有新增终结并晋级的整模型性能轮。
- Deep B32实际fused43/reference0。metadata/cache/index约310ms，其中线程CPU约254ms；不能将CPU全解释成可删除的业务逻辑，也不能把整个区间当成收益。源码确认逐请求selection D2H，建议单HOST TU合并回读，保留全部indexer、因果检查和commit；组件有稳定收益才做一次整模型确认。每一步都有ETA和停止条件，详见JSON。
- Kimi B adaptive初步比fixed总吞吐高5.52%/4.74%，逐用户中位仅3.458/1.750tok/s，>15达标goodput仍为0。完成当前新64题；A完成原33卡低并发短对照。不开第三条Kimi路线。
- GLM首配对由root报告common emitter +16.79%，完整ABBA、质量及终结尚未完成；继续现有719，不能拿stage0吞吐代表整模型。
- E3740的重复历史T64不是当前自然请求瓶颈证明；868/889已否定相应EP2路线，旧CPU提案不再算新路线。大GEMM峰值和逻辑权重字节不能预测请求性能。
- 质量允许token不同；最终依据真实benchmark、自然EOS、任务分数、逐用户时间线及全部卡数吞吐。保留失败证据，修复现有路径/分析器，不增新框架。

详细实验卡、source/MD链、资源、预估耗时与退出条件见同名JSON。
