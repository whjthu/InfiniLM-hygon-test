# Astra review 103

仅消费 910371.558、910376.701、910371.564 三个新合格整矩阵各一次（2 组件 + 1 Model）；不计 CPU、探针或单个子案例。目标尚未完成。关键输入 SHA 全匹配；未改 ledger、共享 Git 或任何 GPU/live 状态。

- **Kimi：推进目标端集成。** 558 的 C32K1 全 query 205.740→116.824ms；564 在已用2D head基线下 C32L512 追加1row 121.587→76.646ms（延迟−36.96%，速率+58.63%），首fill基本持平。二者均为合成 targetfeatures 组件证据，不能相乘成 Model 加速。最坏 hiddenRMS0.0484/最低前缀一致0.4286只作诊断，尚无实际接受率/任务质量结论。
- **先完成冻结421/565。** 当前 PP2TP16 cap16/pool33/pages132 的5.119GiB实际余量仅比4GiB底线多1.119GiB；不能据此推定cap32。CPU精确核算pool65/pages260、state/KV/workspace/临时峰值，再选新PP2cap32 constructor或已知PP4cap32。必须覆盖真实C16/C32，两个C16不能冒充C32。
- **下一次单load验证cache。** 固定head2D、目标权重、state宽度、采样与telemetry，以新cohort/cache reset做OFF/ON，C16与真正C32配同任务/AR参考。append_many一次打包本cycle全部已提交feature；验证epoch/请求ID/past/退场/回滚与最大上下文。当前564 GPU仅到L534；L1024只有CPU覆盖。同步报告接受长度、verify/replay、完整draftRPC、自然EOS正确goodput、用户首末token速率与TTFT/ITL。质量匹配按任务答案，不要求token一致；32题仅pilot。
- **GLM：保留M16。** 701 common20.9252→18.0253tok/s（−13.86%），用户mean0.7559→0.6961；双方31/32正确、32EOS、cap0，15tok/s达标均0。双方PFmicro8，不能将PF差异说成M32收益。CPU审计M>16 lease fallback/分配生命周期（73/176调用，仅假说），找到明确成本后再做一个改动；不为同一问题重新冷load。现有layers hostwall混入异步等待，不足以宣布stage1计算瓶颈。
- **Deep：等唯一1069。** 现有8卡6phase profile完成并审计后，按暴露成本选一个优化；不重复提交、不跑HC/MoE网格。

历史PP4 C32 aggregate30.410→33.012（+8.56%），用户median1.445→1.449（约+0.30%）；C16 pooled16.283→30.212不代表C32性能。PP2与PP4分别披露33/65卡资源效率，拓扑变化仅作部署比较。

分工：kimi_continue_92完成当前Model与cap32精确预算；kimi_state_only_replay_93准备默认OFF的不可变cache集成；deep_continue_92负责GLM CPU归因和现有Deep1069。GPU同资源串行，CPU准备可并行。910384 unrelated loaderchat受保护，不访问。

详细动作、门槛、输入SHA与证据边界见同名JSON。根任务应在implementation103将三矩阵各登记一次并重置新波次计数。
