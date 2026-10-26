# Astra review 115 · 早触发审阅

继续同一 **474/586/277** 的冻结 50 case；不重启、不热改。当前证据不支持新的全模型 GPU。此次只消费 **910375.761 一轮**，不是三轮；部分 C2、CPU 可行性不计轮次。

输入 SHA、指定 10 项源 SHA 全部一致，TP32 source-index 的 37 项内层源也全部一致。已读仓库 AGENTS、协调协议、Astra 和质量政策及对应任务/结果。未执行 Git、SSH、上传、编译、GPU 或仓库修改。BW1100 当前 status 本地缺失，未来执行由 root 再做正常协调检查。

## 判定

- **761 归因成立，但只限组件。** 低 reuse 的未仪表回退约 8.24 ms，仪表差分扰动仅 −0.0078 ms；W2 matrix 同 owner 中位增量约 8.13 ms，quant/reduce 合计仅约 0.058 ms。T8 回退约 0.08 ms 同样保留。不要再用降寄存器、重写 quant/reduce 或盲扫参数解释掉 matrix 主预算。event elapsed 不是 GPU busy，不能据此宣称 bandwidth/occupancy。
- **Kimi 22:32:45 UTC 部分 18 题：** 两臂 18 正确、18 EOS、0 cap；pooled decode 24.6965→25.4363（+3.00%）；实际用户中位 12.9406→13.2889，严格 >15 为 2/18→3/18。首 6 题的 14.75 中位不能代表最新数据。最小采样余量 6,805,258,240 B >4 GiB；first8 screen 通过，仍待全矩阵及退出审计。初始其他请求 prefill 等待必须保留在可见 first-last 内。
- **GLM 755** 只有限定负载任务吞吐 +1.79%、CAPI HOST decode +2.48%；每 epoch 31/32 正确、全 EOS、>15 用户为 0。保留 opt-in。719 deferred feedback、724 prefill pipeline、582 prepared-weight reuse 已做；INT8/bitdecode 负结论不可换名再测。
- **Deep** 64 题两臂 62 正确、全 EOS，用户约 3 tok/s；四节点任务 wall 改善 10.22%、10.84%、−3.12%、18.00%。保留 C4 opt-in 和长输出负例，不能加总独立节点 TPS 当实测服务吞吐。MTP 需要新 head、四流 residual、43 层前缀事务和真实 catch-up；当前 two-position verify 还会失去 q.tokens==1 的 allselected 快路径。

## 最小下一步

| 优先级 | owner / CPU 工作 | 决定与停止条件 |
|---|---|---|
| 1 · K115A | INT8 原 owner + Kimi owner，10–15 分钟：冻结 hybrid 审计，对齐真实 C2/T16 routes、eligible experts、padding 和 FP32 partial/单 reduce 工作量 | 无真实 T16 复用证据就 UNKNOWN/PARK。现 pair 全 routes 最终 BF16 核不可直接用作 subset。当前 33 卡继续；不新增 GPU。 |
| 2 · K115B | TP32 auditor，10–15 分钟：generic I96 已编译对象映射、WORKSPACE_OFF 及 H3/head32 生命周期与同时存活预算 | 当前静态 60.546 GiB 已不满足 4 GiB guard。OFF 约 58.906 GiB 也仅留约 1.094 GiB 给未知开销并保留 guard，绝非 fit 证据。源码或预算不能闭合立即 STOP，不 constructor 探针、不 93 层 load。 |
| 3 · G115A | GLM 原 owner，10–20 分钟：用 755 日志/actual c650 DSO 闭合当前 arrival/ragged 残余关键路径，区分 prefill 阻塞、PP/TP、CPU 提交与 math | 540 旧 M8 和 743 合成 T16 不足代替当前服务 profile。若数据不足，仅准备一次稀疏 current-load profile 卡；先经 root 源码/绑定/资源发布门，当前 0 GPU。不再验证 row2 微收益。 |
| 4 · D115A | MTP source owner + Deep owner，20 分钟封顶：两位置 accepted-prefix 写集、ring 覆盖、valid/length/revision 与 CPU oracle，估计每 row bytes 和 catch-up 调用 | 是 r9287 的增量事务证明，不重复键枚举、不实现完整 MTP。不能原子覆盖 43 层或需要完整历史 replay 则 STOP。无测得 cycle cost，不假设全接受，不开 GPU。 |

未来有 source/CPU 正证据才能提出限定 GPU：Kimi 真实 T16 whole-MoE 一个配对诊断；TP32 完整 provider/H3 预算闭合后才谈真实 32 owner 单层；GLM 只允许能区分残余机制的当前负载 profile。都不由本报告自动启动。完整参数、预热/配对、退出、质量与资源要求见 JSON。

320 卡保留资源不等于 320 卡都有可测任务。最有价值的并行工作是继续现有 33 卡矩阵并做上述有界 CPU 证明；不要假占余卡。已知 Kimi load 1710.53 s，而 761 GPU step 58 s，重复冷加载的代价明显；准备/排队总分解尚未知。

root 应逐卡记录采纳、暂缓或拒绝及理由。所有晋级同时看任务质量、自然 EOS/cap/未完成、实际逐用户速度、TTFT/ITL、严格 >15 goodput 与全部服务卡分母；无 token identity/F64 门，不移动首 token 或靠短输出制造收益。

