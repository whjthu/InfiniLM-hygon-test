# Kimi 投机推理优化实施文档

版本 1  工程设计与证据基线

适用对象：InfiniLM 的 Kimi K3 与海光 BW1000 部署。读者为模型、算子、运行时和服务工程师。本文依据仓库截至提交 94211540 的证据编写；新增内容为实现建议，未执行新的训练、编译、GPU 验证或基准。本文所述方案不自动适用于其他 Kimi 版本。

## 1 实施结论与目标

建议沿用现有 DFlash2 短链草稿，先完成当前 Kimi 运行时的状态事务与真实请求验证，再优化 target 批量验证及部分接受后的重放成本。首版实现 greedy 验证，逐步尝试 K 为 0、2、4、7；默认保持 AR。只有在同硬件、同模型实现、同工作负载的实测收益成立后，才按请求启用投机。

这条路线已有可复用基础：历史真实目标模型与草稿协同生成跑通；部分数学题获得约 2.23 至 2.85 倍的后端 decode 加速。它还没有证明当前最新重构入口、32K 正常代理请求、广泛质量和客户端流式时延的完整收益。自然语言样例也出现过明显变慢。因此，投机长度必须由实际接受前缀与总成本共同决定，不能固定追求最长草稿。

主要目标是提高单请求有效 decode 速度，保留 32K 上下文和普通 4096、thinking 8192 的输出预算。其次考察并发下吞吐、每用户速度与尾时延。超过 15 tok/s 是待验证目标，不是本文承诺；历史最好小样本约 13.90 tok/s 也没有达到它。

交付完成应同时包括：可回退的运行时接口、状态一致性检查、真实草稿契约、质量基准、客户端 token 时间线、正常退出与资源清理证据。只有组件变快或 CPU 检查通过，不算完成。

## 2 现有证据与不可混用的结论

| 证据 | 已验证内容 | 使用限制 |
| --- | --- | --- |
| kimi708 真实投机循环 | 32 卡 target TP16 PP2 加 1 卡 draft；greedy；真实 EOS；两道 GSM8K 正确 | 历史 Native 和短输入；两题 smoke；非当前服务验收 |
| GSM8K 样例 0 | AR 4.908 tok/s，K7 10.964 tok/s，约 2.234 倍 | 有效 decode 为 156 token；不是基准总分 |
| GSM8K 样例 1 | AR 4.875 tok/s，K7 13.904 tok/s，约 2.852 倍 | 有效 decode 为 104 token；无充分重复和置信区间 |
| natural0 的 K7 | 4.561 降至 3.480 tok/s，倍率约 0.763 | 512 输入与 64 输出上限，截断样例，保留作性能负例 |
| kimi558 草稿查询 | C32 K1 草稿查询布局约 1.76 倍组件收益 | 合成特征；不含 target verify；不证明模型加速 |
| kimi531 草稿真批次 | 批处理与资源生命周期机制通过 | 数值等价和 target 接受率未通过资格 |
| Kimi252508 当前长输入 | 8635-token 真实请求下，append512 相比 append8 的 prefill 约 4.07 倍 | 属于 prefill 优化；draft=false，不能记为投机收益 |
| r255 的 C1 T1 时序 | 后端约 4.90 至 4.93 tok/s；host CAPI 时长跟随发射间隔 | 不是纯 GPU 占比，不是可消除开销百分比 |

kimi708 的 best 数字采用后端 emitter 的旧计时定义，包含 draft、通信、verify、重放等；不能直接与新 HTTP 客户端指标合并排名。历史 W8 表示 1 个 anchor 加 7 个 proposal；W16 表示 15 个 proposal，不能写成 K8、K16。当前固定 checkpoint 是 block8，历史 K15 路径不证明该 checkpoint 具有经过训练的 block16 能力。

相关依据为附录的本地记录 E1 至 E7。文档中的新接口、调度和状态快照设计均为建议，需要按阶段验证。

## 3 为什么 Kimi 的投机实现更复杂

投机推理让小草稿模型快速提出多个 token，再让大目标模型一次验证。大模型能够把多个位置并行计算，可能以低于逐 token AR 的成本提交多个正确输出。收益取决于接受前缀长度，以及验证、状态恢复和草稿的合计耗时。

对当前 Kimi，状态不仅是可按长度裁剪的 MLA KV cache。KDA 的递归状态矩阵、因果卷积历史、每层状态行、位置与页表也随 token 更新。验证完整草稿以后，即便拒绝了后半段，递归状态已包含那些被拒 token，不能只把逻辑长度改回去。

此外，target 分散在 TP16 PP2 的 32 卡上。所有 rank 和 stage 必须共同使用同一请求、同一接受长度和同一提交世代。草稿还依赖 target 中间特征，AttnRes 特征来源需要明确。某一层取错特征，或只在 stage0 回滚，会造成隐藏的语义漂移。

因此，实施顺序应是状态正确性、特征契约、实测成本、服务集成。先将 K 拉大再查回滚，通常会同时扩大错误与开销。

## 4 草稿模型的选择与特征契约

### 4.1 固定现有 checkpoint

当前可复用的是第三方 lightseekorg/kimi-k3-dflash2，固定 revision 为 e77935fb4804e17eb55085bffd045eae1d779769。它不是已确认的 Moonshot 官方草稿权重。仓库记录的 config SHA256 为 aac0900105b9099b99a452c728ff9c4685889d264f7bed0101e57788df8f09c8。

已审阅配置为 5 层、hidden size 7168、5 个 target taps 为 19、37、66、78、90，fc_norm=false，MLA 与 SWA 混合结构，block8，最多 7 个额外 proposal，mask token 163592。现有权重约 7.386 GB；独立历史草稿加载约 20 秒。内存准入应按实际权重、workspace、上下文和峰值测量，不以文件大小直接认定能装下。

公开训练 recipe 的 6 层 6 taps 配置与此 checkpoint 不同。精确训练 target revision、量化和历史特征流仍为 UNKNOWN。embedding 相同只说明 embedding 对齐，不说明全部注意力、专家与 AttnRes 参数相同。不能把最新网页 main 当作该 checkpoint 的训练生成记录。[公开配置与模型卡 R2]

### 4.2 特征与位置必须形成可检查契约

每份特征记录包含 target commit、Native SHA、量化与数学 flags、tokenizer 与模板 hash、request ID、token IDs hash、逻辑位置、tap 层、norm 前后、AttnRes stream 类型、consumer 定义和 dtype。TP 分片拼接顺序与 PP stage 来源也应固定。

草稿 query 的 slot0 是已知 anchor，slot1 至 slot7 预测后续 token。上下文特征只允许 anchor 之前已经提交的位置；query 的位置为 anchor_position 加 0 至 7。禁止使用未来真值或被拒分支特征。DFlash2 selector 要按已选前驱逐步形成路径，不能把各行独立 argmax 当作等价算法。[作者实现 R3]

AttnRes 的 completed-layer prefix 与 next-consumer mixture 不是同一张量。现有契约分析支持继续保留当前实现，但没有证明历史训练流。首次只做同一已驻留 target 的小规模、预先定义的特征对照，记录两种来源并在 held-out 请求上比较；不能用同一条训练轨迹挑最有利 stream，也不为此单独重复加载目标模型。

### 4.3 草稿上下文同步

草稿缓存只对应已确认输入前缀。每轮完成后，追加真正提交的 token 对应特征，丢弃临时 mask query 和被拒后缀的 KV。当前草稿若不能导出某个已接受位置需要的真实特征，应使用明确的补算路径；不能用 verify 末尾特征代替中间前缀。

服务必须区分 target 已处理 token 数、用户已发射 token 数、draft 特征前缀长度和 anchor 位置。它们不总相等。跨请求缓存必须按 tokenizer、模板、target 数学实现、adapter 和会话世代隔离。

## 5 单请求算法与位置约定

### 5.1 推荐的 pending anchor 约定

设 target 已提交状态只处理了前缀 x，长度为 n。anchor z0 已由 target 选出，可以已经发射给用户，但尚未进入 target 状态。draft 以 x 的已提交特征与 z0 为条件，提出 z1 至 zK。

target 对输入块 [z0,z1,…,zK] 做一次 causal verify。第 i 行输出分布 p_i 预测该行输入之后的 token：p_0 用来验证 z1，p_1 验证 z2，以此类推，最后一行 p_K 生成全接受后的 bonus token。所有行的 mask 和 position 必须与逐 token AR 一致，不能让一个候选看到未来候选。

greedy 模式下，依次要求 z_i 等于 argmax(p_(i−1))。遇到首个不相等即停止，不能跨过拒绝位置继续接受。若接受了 a 个 proposal，实际保留的 verify 输入是 [z0,…,z_a]，长度 r=a+1；下一个 pending anchor 从 p_a 选出。a=K 时，p_K 给出 bonus。

最容易混淆的是 r=a+1 包含上一轮已经选出的 anchor，而不包含新选出的 correction 或 bonus。新 token 的用户提交集合是 z1 至 z_a 加新 anchor，最多 a+1 个；target 状态前进的长度也是 a+1，但具体 token 集合不同。不能把 correction 写入未计算它的缓存。

例如 n=100、K=7，z1 和 z2 接受、z3 拒绝，则 a=2、r=3。验证临时状态处理了8个输入位置，但应从原 active 重放 z0、z1、z2，只提交到长度103；从 p_2 选出的 correction c 成为新 anchor。用户这轮新增收到 z1、z2、c，下一轮必须从长度103的状态消费 c。如果一个 proposal 都不接受，只消费旧 z0，再发出 correction；全接受时消费完整8项并发出7个 proposal 与 bonus。K0 直接跳过草稿，执行单位置 target forward。

### 5.2 伪代码

```python
# 设计伪代码 不是现有可直接调用的接口
state, features = target.prefill(prompt)
anchor = target.select_next(state)  # 尚未被 target 消费
emit_once(anchor)
while not stopped:
    K = choose_K(request, remaining_budget, observed_costs)
    proposals = draft.propose(features, anchor, K)
    txn = target.begin(request_id, epoch, state)
    verified = target.verify(txn, [anchor] + proposals)
    a, next_anchor = greedy_prefix(verified, proposals)
    retained = a + 1
    target.prepare_retained_state(txn, retained)
    target.commit_all_owners(txn, retained)  # 成功后才发射新输出
    features = update_from_committed_prefix(features, verified, retained)
    emit_until_stop(proposals[:a], next_anchor)
    anchor = next_anchor
```

prefill 首 token 与 decode 首轮的时间边界应写清。示例省略的 EOS、取消、异常处理必须按下一节实现；任何提交不确定都不能直接重新调用上述循环。

### 5.3 EOS 与输出上限

anchor 已是 EOS 时不启动下一轮。如果接受前缀内出现 EOS 或 stop，立刻截断可发射后缀；拒绝的 EOS 不得终止请求。correction 或 bonus 命中 EOS 时同样结束，并禁止下一轮消费它。

K 不得使可发射 token 数超过剩余输出预算。内部 mask、被拒 proposal、重放输入、padding、EOS 特殊符号是否计入有效输出须按服务约定统一；有效吞吐不计填充和拒绝 token。stop 字符串可能跨 token、跨投机块，应交给同一增量解码器处理，不能每块独立查字符串。

## 6 KDA 与 KV 状态事务

### 6.1 首版沿用 active 和 shadow 状态

为每个逻辑请求分配 active 与 shadow 物理状态行。active 保持已提交前缀，verify 从 active 读取并把完整 W=K+1 输入的结果写入 shadow。MLA KV 必须避免覆盖仍被 active 引用的页，至少保证被拒范围不可访问；KDA 矩阵和卷积历史分别维护写行。当前两行状态逻辑可参考 kimi_pair_state_r9006.hpp，但该历史组件有 width≤8、past≤1024 等限制，不可直接视为 32K 生产实现。

接受全部 proposal 时，shadow 正好对应整个 verify 输入，经过所有 owner 确认后交换 active 与 shadow，不重放。仓库的历史逻辑已有全接受免重放分支，应复用，不能作为新增收益。

部分接受时，从仍未改变的 active 状态出发，只重放 retained 前缀，重建正确 KDA 和卷积状态后再提交 shadow。MLA 页表和有效长度一并推进。重放是首版正确性路径，成本可能很大，但比未经验证地裁剪递归状态可靠。

### 6.2 状态机与原子提交

建议状态机为 READY → VERIFYING → VERIFIED → PREPARING → COMMITTED。错误时进入 ABORTED 或 POISONED。epoch 和 request generation 单调递增；禁止旧响应、重复提交、逻辑请求行重用时的迟到消息、read/write 同行及跨请求共享 shadow。

接口语义建议如下。函数名是新设计，不代表已有 C API。

| 接口 | 关键输入 | 成功后的保证 |
| --- | --- | --- |
| begin | request generation epoch past active shadow | 保留 committed 状态并验证容量 |
| verify | 带 row offset 的输入块和 causal metadata | 输出所有验证行及临时状态 不发布 token |
| prepare | 每请求 retained 长度 | 每个 owner 获得可提交的对应前缀状态 |
| commit | generation epoch 所有 owner 确认 | 各 stage 推进同一逻辑世代 |
| abort | 最后可证 committed 世代 | 清理临时页和行 未证明可恢复则隔离请求 |

分布式 commit 不能只发一条无确认广播。先收集全部 owner 的 prepare 成功，再通知 finalize；保留可恢复的旧映射，直到提交协议结束。通信中断后可能出现部分 owner 已 finalize，应通过事务日志和世代对账恢复；不能假设 rollback 总能原子完成。无法确认一致性时隔离该请求、保留诊断，不在混合状态下回退 AR。

token 只在全局提交确定后发射一次。客户端断开、超时、Native 失败都要标记事务终态，避免下一次复用该行时继承脏状态。安全回退 AR 只允许从明确的 committed 边界开始。

### 6.3 第二阶段减少重放

候选一是在 verify 时输出每个候选前缀的 KDA 与卷积快照，从 a 对应的边界直接选择状态。候选二是在块内保存稀疏检查点，部分接受只从最近检查点重放。候选三是只计算状态相关层的前缀恢复，但必须证明所需输入激活可用且与原路径一致。

不要尝试从块末状态简单减去被拒 token。KDA 的门控、递归和舍入使逆变换并不可靠；卷积尾部也需选取正确的历史窗口。所有快照方案先与 active-shadow 重放路径对照，再谈性能。

快照内存按实际各卡局部形状估算：M ≈ 并发请求数 × 快照边界数 × 本地KDA层数 × 每层状态字节，另外加卷积、页表和对齐。若每层局部状态形状是 H×Dk×Dv，则 BF16 矩阵为 2HDkDv 字节；不同 TP 切分不应沿用旧 H6 D128 组件常数。预算还要包含 active、shadow、workspace、draft 和分配器峰值。

## 7 目标模型批量验证

target.verify 必须是真正的多位置计算，不能在 Python 内循环 K+1 次普通 decode 再称为投机加速。数据面应按请求组织 token、past、offset、width、read row、write row、page table 和 generation。

KDA 要支持非零历史状态的 chunk forward，返回各验证位置需要的隐藏值及可恢复状态；卷积要确保每个 token 的历史窗口正确；MLA 的多 query causal mask 不能读未来 token。MoE 的路由、专家调度、TP collectives 和 PP 传输都需覆盖 W>1。head 输出正确的每位置分布；greedy 可优先在 device 上返回 argmax IDs，减少完整词表回传。

top-p 等随机保分布采样需要额外概率信息，不能沿用 greedy 只返回 IDs 的契约。仅裁掉未使用的 head 行可能降低草稿成本；不应随意减少经过训练的 8 个 transformer query 位置，否则 attention 路径发生变化。

原 32K cache capacity、状态行数量、token workspace T 和单请求 width W 是四个独立限制。例：若实际 workspace 仅支持 totalT64，C8×W8 可放入一波，C16×W8 要分波；W16 时同一上限只容 4 请求。该例是容量算术，不证明任何当前 Native 已支持这些尺寸。

## 8 调度 通信与服务接入

### 8.1 先独立草稿进程

初版建议复用历史 32 卡 target 加 1 卡 draft 的隔离结构，减少 target 显存争用。总资源分母为 33 卡。target 的本地所有 owner 结束计算后，由明确的 feature owner 合并需要的 taps，再把增量特征送给 draft。每轮只传新增 committed 特征，避免重复发送整个前缀。

协议应包含 request ID、generation、epoch、committed length、anchor ID、tap 契约 hash、dtype、shape、payload bytes、checksum 和 deadline。验证完整 header 后才分配和解包。短写、半包、重复响应、断连重连均用同一协议处理；不要只靠 TCP 顺序推断请求身份。

projected context 缓存可作为后续草稿优化，优先降低重复投影与传输；每一次缓存命中都必须绑定相同权重与 feature stream。不要把 loopback RPC 组件时延当成跨机网络时延，也不能把 query wall、RPC wall 等嵌套时间相加。

按五个7168维 BF16 tap 估算，每个位置的完整特征为5×7168×2=71680字节，约70KiB；若按15位置每秒传输，原始增量约1.08MB/s，未计协议开销。这是形状推算，不是测量值。单请求下更应测量同步次数、打包与往返等待，而非只归因网络峰值带宽；长前缀初始化仍可能很大，不能每轮重传全部上下文。

### 8.2 按实测收益选择 K

对 K=0、2、4、7 维护平均提交 token 数、验证时长、重放概率与时长、draft 往返、队列等待。按请求类别、上下文长度和并发分桶，预测下一轮每有效 token 成本；选择最便宜的 K。样本不足或质量尚未资格时保持 K0。

动态退回 AR 的条件可包括：连续低接受前缀、预测收益低于已预设最小幅度、draft 超时、剩余输出很短、target workspace 紧张或排队恶化。首版采用保守分桶策略，待稳态证据充分再加复杂预测器。策略在测试集冻结，避免用同一评测数据调整阈值。

### 8.3 流式服务

投机块会突发提交多 token。SSE 应按原始顺序送入同一 reasoning、content 和 tool-call 解析器；不能让一个块把 reasoning 与 content 混合写错字段。多 tool-call ID、反向返回的工具结果、长历史保留、取消和 EOS 都要复用正常代理测试。

客户端时间线记录每个有效 token 的到达事件和单调时钟。一个 SSE chunk 携带多个 token 时可记录同一到达时间，并另外报告 chunk gap；不得人为在这些 token 之间均匀插值。P50 ITL 为零可能只是突发输出，不能据此宣称平滑流式。已有直到完成才缓冲返回的 HTTP 路径不能用于客户端逐 token 时延资格。

## 9 随机采样与质量边界

第一版建议 greedy 前缀接受，减少未定义 proposal 分布导致的错误。若要求与 greedy AR 一致，应在相同目标实现下验证多位置 logits 的因果性；不同 shape 或量化路径可能带来舍入差异，应显式记录。用户允许 token 不逐位相同，不等于允许状态错误、未来信息泄漏或隐藏质量下降。

精确保留目标采样分布需要真实 p 和 proposal q。常规投机采样对候选 z_i 按 min(1,p_i(z_i)/q_i(z_i)) 接收；首次拒绝从归一化的 max(p_i−q_i,0) 采样；全接受时再从 target 分布采 bonus。此处 p_i 表示正确已接受前缀下的目标分布，符号下标与第5节 logits 行索引需在代码中转换清楚。[原始论文 R1]

DFlash2 selector 会改变 proposal 生成规则，原 unary logits 的 softmax 不一定等于最终 q。若采用确定性 greedy draft，可把 q 解释为对应候选上的 delta 分布，但要实现适配该 proposal law 的接受与校正；不能只将 stochastic target 的 argmax 与候选比较。temperature、top-k、top-p、repetition penalty 和 grammar mask 的真实分布变换也必须纳入推导。

若使用 target top-k 内即接受、阈值接受或语义近似接受，应明确列为改变分布的独立实验路线，用任务基准负责质量。不能同时叫 lossless。算法保分布也只针对所绑定的量化 target 实现，不代表恢复未量化模型的原始分布。

## 10 性能模型与达标条件

一轮近似成本为 Tcycle = Tdraft + Tfeature_transfer + Tverify + Trestore + Tcommit + Tservice。设 A 为一轮接受 proposal 数，理想稳态平均有效输出约为 E[A+1]；遇到 EOS、stop、预算末尾时应按实际有效数计。

Rspec = 总有效 decode token 数 / 总 decode wall。粗略盈亏线是 E[A+1] > Tcycle / TAR，其中 TAR 是同配置普通 AR 的每 token wall。若目标为大于 15 tok/s，则需实际达到 Tcycle / E[有效输出] < 66.67 ms。不要拿两个不同实验的均值相除直接估计加速。

若假设每个 proposal 的条件接受率恒为 α 且各位置遵循相同简化模型，则 E[A+1] = 1+α+α²+…+α^K。真实 DFlash2 接受具有位置和前缀依赖，应直接统计 P(A≥j) 并用 1+ΣP(A≥j)，不能默认独立。

历史 GSM8K1 的 K7 平均每轮约 6.5 个有效输出，完整 decode 约 13.90 tok/s；自然语言样例的 K7 每轮仅约 1.91，反而降速。该对比说明“同一个草稿、同一个 K”可有相反收益。回放约占数学样例 decode 的 29% 至 30%，是值得测的候选，但将它归零得到的反事实速度不是实测优化结果。

组件 head 加速、MoE 加速都需按实际 wall 占比折算。若 draft 只占总成本 5%，即使 draft 完全消失，整体上限也只有约 1/0.95=1.053 倍，这是说明性算例，不是本机实测占比。当前 C1 T1 host CAPI 的约99%描述性比例也不能当成 Amdahl 可删除比例。

## 11 验证矩阵与测试实现

### 11.1 CPU 契约测试

先对每个 a=0 至 K 验证 accepted、retained、下一 anchor 与缓存前缀关系。K0 是普通 AR；全接受无重放；部分接受仅重放原验证输入的前 r 项。覆盖错误 owner、旧 epoch、越界 row、重复请求、乱序 offset、read/write 别名、超容量、取消和重复 commit。

使用可追踪的递归 toy state 验证拒绝后不能仅裁长度；确认重新构建的状态与逐 token 参考一致。此类测试是状态算法验证，不拿 mock 的运行成功宣称真实 Native 通过。

### 11.2 真实算子与分布式验证

在同一实现下比较 W1 与 W2、W4、W8，历史位置覆盖 0、短前缀和实际长前缀。对 KDA、卷积、MLA、非零状态读写、页边界、32K 边界、ragged 行映射、TP 与 PP 的不同 stage 分别核对。将每个接受前缀通过“重放参考”和“候选快照”继续生成，检查后续结果及恢复状态，不能只对比一次 logits。

早期允许合理的浮点误差和非逐位相同输出，必须报告误差范围、首处分歧、EOS、异常重复与最终任务答案；不引入 GPU F64。验证数值不稳定时，先隔离 shape、路由或状态原因，不用近似接受掩盖。

### 11.3 真实请求质量

建议首轮冻结至少200条混合任务，分为中文问答、英文数学、代码、工具调用、长上下文与写作；训练、调参和 held-out 分开。后续扩展完整 GSM8K、MATH500、HumanEval 或 MBPP，以及真实工具参数校验和长历史检索。代码执行在隔离环境内。

每个模型版本给出 AR 与 speculative 的分数、样本数、置信区间、拒答、截断、失败、非自然 EOS 和输出长度分布。可预先设定质量允许下降界，例如绝对1个百分点，但它是待项目确认的验收建议；样本不足以排除下降时结论只能为证据不足。两道数学题不能宣称无明显掉点。

### 11.4 性能与服务

同一次加载内完成 AR 与候选，先独立预热，再对冻结请求做 ABBA 或随机配对。建议至少5个完整配对块，并根据置信区间决定是否证据充分；不能重复到出现有利结论就停止。上下文按短、中、8K以上和近32K分层，输出预算保持原值，截断样例单列。

分别报告 C1、C8、C16、C32 在真实支持的 workspace 下的每用户 decode、TTFT、ITL P95/P99、总体吞吐和每卡吞吐。在线测试记录实际到达轨迹、排队和失败，静态批次不能冒充服务并发。

客户端口径采用 Ruser=(n−1)/(t_last−t_first)，n为有效 token 数。prefill、首 token 产生与客户端首包另列，EOS及特殊控制符计数规则固定。服务端 token-ready 与客户端到达分开报告；跨主机只比较本地持续时间，不直接相减未经校准的单调时钟。

### 11.5 清理与故障

所有 owner 正常关闭、请求状态 reset、draft cache 清空、队列取消和 TP PP 同步均有逐 owner 证据。再以独立后检查确认 KFD为空与端口可绑定。强制 kill 后空闲只能证明清理，不能证明正常 Native 生命周期。

启动前应在实际镜像、实际工作目录和同一挂载集合内验证静态依赖；登录节点上 hash 通过不足以证明容器内可见。GLM810 与 Deep1212 的容器路径失败是现成教训；该经验用于本路线，不把其他模型 CPU618 的检查直接迁移为 Kimi 启动资格。

## 12 分阶段实施与退出条件

| 阶段 | 主要工作 | 交付与进入下一阶段的条件 |
| --- | --- | --- |
| P0 基线绑定 | 固定当前 Kimi Native 量化 模板与 AR 指标 核对旧投机差异 | 证据和去重清单完整 无新增性能结论 |
| P1 状态事务 | CPU状态机 W1与W8因果验证 active shadow及部分重放 | a0至K与异常路径通过 原状态可恢复 |
| P2 真实短链 | 复用block8草稿 greedy K0 K2 K4 K7 | 同驻留真实EOS 质量初筛 端到端配对 |
| P3 减少重放 | 每前缀状态或稀疏检查点 二选一 | 相同接受轨迹下状态正确且实际wall下降 |
| P4 草稿与调度 | projected context 增量传输 K自适应 | 不降低held-out接受效果与服务尾时延 |
| P5 服务与交付 | 最新入口 多轮工具 真流式 全面质量 | 质量及客户端指标通过 默认可回退 |

优先完成 P1 和 P2，先得到真实轮次的接受长度分布与各项 wall。如果 verify 主导，先优化多位置 KDA、MoE 或 TP PP 临界路径；若重放主导，先实现可选状态；若草稿主导，再做草稿缓存、head 或通信。不要同时更换量化、草稿、tokenizer 和 target 状态策略，避免无法归因。

停止或回退条件包括：状态不一致、无法确认事务世代、后检查残留、质量明显下降、同负载稳态无收益、尾时延超过预设容忍范围、显存低于原安全余量。单次尝试、资源与失败重试政策由实际实验范围约定；本文不创建执行授权。

准备和排队耗时、首次加载、测量、质量检查应分别记录。工程工期取决于真实接口和训练栈，当前无可靠依据承诺几小时完成。首个决策应来自小而完整的真实配对，而不是无限扩大验证清单。

## 13 更长草稿与重新训练

现有 block8 不能只改配置成为训练合格的 block16。只有在 block8 的状态、验证成本和接受率已经量化后，才评估更强草稿。先固定与当前 target 一致的特征流、量化数学实现、模板和 on-policy 数据，保留独立 held-out 请求。

可先继续训练当前 block8，再比较 block16。对 backbone、unary 和 selector 的训练范围应明确；只调 selector 或扩大 query 数，不能代替后半块的监督。训练目标除 token loss，还应评估自条件 rollout 的接受前缀、真实 decode wall 和任务质量；teacher-forced 逐位置准确率不等于在线接受长度。

训练显存须单独预算。约3.69B BF16参数本身约7.39GB；若参数2字节、梯度2字节、Adam一二阶状态各4字节，核心存储约44.3GB；FP32 master另约14.8GB，尚不含激活和workspace。冻结共享权重、梯度检查点和分片可改变该预算，但不能默认一张64GiB卡可训练。数据生成与训练卡应错峰使用，避免长时间占住32卡 target 而没有可测任务。

EAGLE3、树验证、PEARL 式重叠可登记替补，当前不作为最短路径。树需要分支KDA状态与树mask；重叠需要双缓冲和事务取消。已有串行耗时中 draft 很小时，隐藏 draft 的收益也有限。没有契约匹配和实测支持前，不承诺公开论文倍率可迁移。

## 14 建议的代码组织与交付清单

以下为建议的新模块，不代表已实现。

```text
runtime/speculative/kimi/
  contract.py           checkpoint tap tokenizer math identity
  draft_worker.py       resident draft and incremental context
  protocol.py           request generation epoch and framing
  scheduler.py          K selection admission and AR fallback
  verifier.py           target block outputs and prefix decision
  transaction.py       active shadow prepare finalize abort
native/kimi/speculative/
  state_transaction.hpp KDA conv KV metadata and owner guards
  verify_forward.cpp    causal multi-position actual forward
  prefix_state.cpp      optional selected-state or replay backend
tests/speculative/
  test_anchor_alignment.py
  test_state_transaction.py
  test_ragged_and_cancellation.py
bench/speculative/
  run_paired.py         frozen workload and complete token timeline
  audit_results.py      quality lifecycle and same-scope comparison
```

验收包应包含源码 commit 和 dirty patch hash、checkpoint revision、config 与权重清单hash、各 Native 与依赖hash、训练与feature provenance、硬件与全部卡数、TP PP、输入和输出预算、K策略、每轮接受与重放记录、有效token和EOS、服务端与客户端时间线、质量与清理结果、默认开关与回退说明。

单轮建议记录 request_id、generation、epoch、past、K、A、retained、有效新输出数、结束原因，以及 draft、feature transfer、verify、restore、commit、queue 和 service 的本地持续时间。完整词表、原始激活和巨型trace留实验端，只提交必要汇总和hash引用。

完成标准是：当前 Kimi 部署下，真实请求的质量、状态生命周期与服务行为通过，且实测单用户速度或约定的系统指标有稳定改善。若只某一类任务受益，应按类启用并报告覆盖率，不能宣传全局加速。

## 附录 A 本地依据

路径均相对本仓库根目录。它们是本次文档使用的证据入口；历史硬件与源码版本不得合并为一个实验。

- E1 results/bw1000/2026-10-25/kimi708-dflash2-real-speculative-cycle.json
- E2 analysis/bw1000/kimi-draft-contract/REPORT.zh-CN.md 及 source-manifest.json
- E3 analysis/bw1000/Kimi-strongdraft-longblock-research-r162.md
- E4 analysis/bw1000/Kimi-speculative-latest-r8892.json
- E5 results/bw1000/2026-10-26/kimi531-truebatch-draft-qualified-mechanics-numeric-risk.json
- E6 results/bw1000/2026-10-26/kimi558-draft-flat-head-query-qualified.json
- E7 analysis/bw1000/Kimi-current-C1T1-decode-critical-path-r255.json
- E8 code/bw1000/batch-concurrency/Kimi-speculative-batch-private-r9122a/kimi_pair_state_r9006.hpp
- E9 code/bw1000/batch-concurrency/Kimi-speculative-batch-private-r9122a/batch_transaction_r9122.hpp
- E10 results/bw1000/2026-10-31/Kimi252508-full-agent-history-multicall-paired-gain-and-cleanup-PASS.json
- E11 analysis/bw1000/Kimi-append512-device-body-source-review-r246.json
- E12 analysis/bw1000/Kimi-draft-prefix-calibration-decision-r8669aae.json
- E13 coordination/QUALITY_POLICY.zh-CN.md 与 coordination/PROTOCOL.zh-CN.md

## 附录 B 外部原始资料

- R1 原始 speculative sampling 论文 Fast Inference from Transformers via Speculative Decoding https://arxiv.org/abs/2211.17192 。用于第9节算法性质，不用于预测本机加速。
- R2 固定草稿checkpoint https://huggingface.co/lightseekorg/kimi-k3-dflash2/tree/e77935fb4804e17eb55085bffd045eae1d779769 。本次网页读取受限；配置细节依据仓库已保存的审阅记录E2，不能称为本次重新下载确认。
- R3 作者DFlash2实现 https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/models/dflash2.py 。当前网页已查阅；本地契约以E2保存源码与hash为准，main会变化。
- R4 作者AttnRes捕获说明 https://github.com/lightseekorg/TorchSpec/blob/main/docs/kimi_k3_attn_res_capture.md 。历史审阅入口见E2，不作为固定checkpoint训练运行的独立证明。

作者说明：本文为项目技术方案整理。新增伪代码与目录用于实施讨论，不包含可直接提交的GPU启动命令；未增加任何已验证性能结果。
