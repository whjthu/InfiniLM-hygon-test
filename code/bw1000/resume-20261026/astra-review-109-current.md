# Astra 109 审核

输入 b3993e29 及15个固定源 SHA 已核对。108 四张卡均有落实：GLM724 整模型完成时间下降8.461%，但首微批 TTFT 增51.90%；Deep1099 证明首次初始化不能充当稳态310ms瓶颈，并定位B32 bank调用区间；KimiA33卡完成116例/arm，C2仅20、C4缺失，原审计与独立审计均0，未发生B型OOM。W2/W4负结果，停止宽度wrapper。

本次只消费724、1099、461+576三轮，CPU413及审计不计轮。完整执行卡见同名JSON。

- D109A：立即运行已就绪CPU413的8卡组件候选。原copy已异步，候选仅省view/descriptor/submit成本；B32 warmed wholeCSA稳定≥5%且EOS无回归才进D109B整模型。
- K109A：复用flag3现有host BF16 logits，每行只gather提议/target-top1两值；单epsilon=0.10作为明确的探索参数，OFF/epsilon0原greedy，EOS/结构token严格，正确前缀提交/rollback。先CPU语义检查，随后一次A33卡配对pilot；不造native API、不照搬Medusa参数。
- K109B：pilot正向后同次加载做多域heldout质量、EOS、响应长度校正和实际任务goodput。不同tokens允许；小样本通过不代表统计非劣或普遍5–10倍。
- G109A：单chunk32 CPU实现语义闭合后可执行一次16卡Model配对。固定32次真实计划到达、maxresident16、P/D独立CAPI、全epoch不复用EOS行。报告排队TTFT和预填充插入造成的ITL，不能称HTTP服务。

BW1000物理峰值仍未知；T8约0.3秒的条件周期算术与2.67–3.27ms bank实测局部区间均不是硬件峰值。准备/排队/加载/测量耗时分列；不加框架、无关门禁或Git等待。

