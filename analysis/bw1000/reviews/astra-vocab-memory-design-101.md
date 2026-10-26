# Kimi TP16 词表分片与显存设计审核（独立设计 101）

结论：建议先做 TP16 原 BF16 head 对比词表分片＋完整 logits allgather 的独立组件实验，通过后再做只保留分片权重的私有 Model。现有 DualVocabHead 不能提供显存节省。此报告不是正式第 101 轮三轮性能 review；未消费 review100 ledger，未提交 GPU，未改源代码或 live 作业，仅生成这两个报告。

## 已确认的证据

本地五份 head 源文件的 SHA256 全部与 `Kimi-TP16-head-CPU-compact-r8669z.json` 的 fresh_MD 匹配。其 `head.o` SHA 为 `a7064350a7a4e60ce75f47e057a95100043ed0c5520bf6c360eddf02315097b1`，与 r9122c 成功 CPU receipt 的 linked-object 表一致。最终 DSO SHA 为 `9647453d13f5e56b81fda3a6ec699f5329a83157a294e7e8e2dcffeeb4775e61`，与 PP2C16 manifest 一致。

注册工厂 → KimiK3ForConditionalGenerationHeadR8341 → TextCausalLMHeadR8341 → 最后 PP stage 的 DualVocabHeadR8341。Dual 继承 ReplicatedLinear，构造函数先构造完整 head；TP16 随后直接返回，不建 candidate，policy=1 会报错并复位。TP8 双存储也不能作为节省证据。相关位置：dual 26–55 行、causal LM 27–28 行、model cpp 文件尾注册。

这是强静态链接证据，仍应在下一个隔离实验记录每 owner 的实际类型、权重形状/字节、rank 和 DSO 哈希，以闭合运行时分配证据；不需要碰 live。仅有 RankWorker sizeof 一致不能证明新 Model/模板 ABI 一致。

PP2C16 constructor 已完成，随后 allowner reserve 检查失败，发生在读取权重前；tail 空闲 3.068359375–3.083984375 GiB，各 owner 正常 shutdown。这不是 OOM，也不说明 Model 不可行。PP4 的 01:07:28 UTC 安全快照仍在 load，all64 preload floor 32.0295 GiB，尚无新性能结果。

## 最小实验与验收

1. 在新授权的空闲 TP16 两节点资源上运行独立组件，避免与受保护任务争抢 GPU。A 为现有 ReplicatedLinear BF16；B 为现有 VocabParallel 的 BF16 shard、allgather、行序重排、contiguous 和 sync。输入 M=1/4/8/16/32/64、K=7168、N=163840。优先同一实际 head 权重及实际 hidden；若只能 synthetic 必须清楚标注，不能声称 Model 质量已通过。
2. 同一 load、相同所有 rank hidden，逐 M warmup，至少 30 组配对 ABBA；计 max-rank 主机 wall 与设备 stream 的 GEMM/collective/reorder 分段，包含必要 fence，给 p50/p95 和配对不确定性。屏障在计时区外；CPU 输出拷贝单列。组件同时保留 A/B 可用于速度对照，必须另开 A-only/B-only 新进程测显存，不能把双存储误作 Model 节省。
3. 验证完整 BF16 logits 形状、有限性、词表顺序、每个 shard 边界，记录误差分布、top-k、top1 margin 与差异。先确认输出合理，再做固定质量 benchmark。允许不同 token，不恢复 bitidentity/F64 门。数值阈值按代表性 baseline 变动预先登记；未知误差或 rank 错位直接 NO-GO。
4. 记录实际 IBext_v8 选路、TP16 communicator、collective 时间、每 owner 的 free/used/allocator live/reserved、load/warm/M64 峰值；plugin 名不等于 GDR。性能采用必须有超过噪声的完整 head wall 改善；若仅为降低卡数而接受小幅 head 回退，必须明确权衡并继续完整 Model throughput/card 判定。

## 私有 Model 的最小修改

新建 shard-only head（ColumnParallelLinear 派生或组合）与新的私有 causal-LM 模板，最后 PP stage 直接绑定；不继承 ReplicatedLinear、不保留 full GPU clone。保持 `language_model.lm_head.weight` 参数名与现有完整 BF16 logits CAPI。替换当前 head.o，冻结其余对象；以 .d、nm、link.map 找出所有使用变更具体类型的 TU 并一致重编，使用新类/模板名避免 ODR 混用。独立 DSO 中注册必须唯一；旧 policy setter 对新路径应明确 unsupported，不假装还能无代价切回完整 baseline。

loader 是关键门：现 Python loader 传全局 CPU BF16 [163840,7168]。必须证明 ColumnParallel Weight 只将 `[tp_rank*10240:(tp_rank+1)*10240]` 复制至 GPU，避免先全量 H2D 再切片、转置或双重分片。若现有 loader 做全 GPU 暂存，私有路径改为有界 CPU 切片与明确 local-shape 接口。CPU 源缓冲存活至 copy fence。验证每 rank shard 的来源、无缺行重行与 key 覆盖。保留 eager-only、Runtime owner、CCL 借用寿命和异常 drain。

完整 head 2,348,810,240 B（2.1875 GiB），TP16 shard 146,800,640 B（140 MiB）；每 tail owner 静态节省 2,202,009,600 B（2.05078125 GiB）。按旧 tail 最低值投影，空闲约 5.119140625 GiB。该数字尚未扣新 workspace/collective/allocator 峰值，绝不是实际容量证明。M64 gathered 与 contiguous 同时存在约 40 MiB，另有 local logits 1.25 MiB；不能只计最终 20 MiB。

Model GO 必须实际验证所有 owner 在 constructor、loading/finish、warmup、最长支持上下文及 workspace 峰值仍严格 >4 GiB，并完成 create/load/forward/destroy/post-empty；不降低门。之后测 PP2 target32+draft1 对 PP4 target64+draft1 的同类请求总吞吐、每卡有效输出吞吐、请求延迟与真实用户 15+ 标准。质量需无明显下降。head 加速不等于最终 Model 加速，PP2C16 与 PP4C32 并发不同必须明示。

## 后续边界

Distributed argmax 另立后续：当前 pair/batch flags=3 强制完整 BF16 logits；旧 token-only flag 仅避免 CPU copy，并没有免掉 head/allgather。新 token-only 接口需全局 vocab offset、tie/NaN 规则、logit processors、全 rank token 分发、EOS/commit/replay 验证；top-p 等随机采样与需要分布的 speculative acceptance 不能用 local argmax 代替。M64 完整 logits 为 20 MiB；若 ring allgather，每 rank 接收有效 payload 18.75 MiB、发送量同级，实际跨节点 wire 流量由算法拓扑决定，须测量。无需让这项阻塞 first fullgather 实验。

Cache decouple 只作备选：C16 physical32/pool33、C32 physical64/pool65 与 speculative shadow/replay 状态有关，不能盲减。先按 cache kind/layer 量化高水位，证明 slot 生存期、rollback、最长上下文与分配策略。Runtime trimMemory 以及 PREPROCESS/prepare/reset 调用已经存在，不能再当新优化。

硬件分析使用 `2*M*7168*163840` FLOPs，分片每 rank /16；权重读取从2240 MiB降至140 MiB，流式算术强度约 M FLOPs/byte。结合真实 shape 的 BF16 实测、HBM 字节、collective/reorder、PP bubble、MoE 路由与 cache 才能解释性能。DenseTF 或纯 read-HBM benchmark 不是 Model physical peak；没有验证芯片峰值规格就不输出数值 physical roofline。最终度量以有效接受输出 tokens / 总参与卡数（含 draft）/ wall time 为准。

详细字段、证据 SHA256、GO 条件与风险见同目录 JSON。受保护作业 910384、910375.727、910381.487、910371.541、910376.694、910374.1065 均未访问或修改。
