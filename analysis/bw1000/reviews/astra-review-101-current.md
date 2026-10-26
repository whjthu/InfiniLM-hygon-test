# Astra 正式性能审核 101

输入 SHA256：`37eee98941767e9108a799b64b244d32dac861c5bc575df90a7fc879484bfc39`。13/13 冻结来源 SHA 全部通过。仅本地读证据并写报告；未操作 GPU、live/source 或 ledger。此前词表设计 101 独立，不消费正式轮次。

本次只消费 **910373.3483、910376.694、910374.1065** 各一次：host RDMA 矩阵一轮、GLM Model 对照一轮、Deep 因果 profile＋同驻留 B32 一轮，共三轮。CPU 工程、pre/post、in-flight416、未完成 PP4 均计零。三 handle 均在 unreviewed 且不在 reviewed watermark；root 实施记录原子记账并将待审核计数归零，不重复拆算矩阵。

## 判定

|对象|证据结论|行动|
|---|---|---|
|GLM694|同驻留 common decode：C16 16.6471→17.6055（+5.757%）；C32 20.6960→22.3935（+8.202%）。真实 ragged 分支命中，terminal/audit/post695通过。|保留已验证 resident-KV B<=16 优化；推进 B32 独立组件后再一轮 Model M32。|
|Deep1065|真实4题1681token，344行×32marker，attention inclusive 62.577%、MoE envelope34.395%。B32 fresh32为33.3616 tok/s、4.1702/card，32/32正确EOS。|优先 attention 内部分解；不能拿不同旧 full64 样本计算新 gain。|
|Kimi|727/487/541真实C16/C32仍待完成；416 head组件进行中；Kmax CPU543通过。2题重复 smoke 的 AR55.46 vs state48.93 不是 benchmark。|保留运行队列；并行推进 head shard 与 draft head/selector maxKs，不能由95.9%接受率宣布加速。|
|host RDMA3483|四rail 58.52–66.96 GB/s；CPU仅0..7，部分rail非NUMA本地，wait计数饱和。|只作host-memory观察；接续已编译raw InfiniCCL AR，非GDR/Model峰值证明。|

GLM试点 MMLU 两臂29/32；GSM C16两臂16/16、C32两臂31/32，全EOS、cap0。C16/C32 ON用户平均仅1.461/0.785 tok/s；Deep用户平均1.120、中位1.031、最大2.082，均未达15+目标。小样本无观察下降不是正式非劣证明。GLM每C只有一对，不能扩张为跨场景稳定提升。

review100 的 GLM集成与实际对照、Deep既有owner API测量已落实。Kimi原722/482构造前失败已保留；当前分区为实际[0,24,47,71,93]，继续当前运行，不为审核重启。Deep event elapsed包含host gap/依赖等待，嵌套span不能相加；ON/OFF均值比1.00325只是带方差的bracket，不是严格因果 instrumentation overhead。独立8owner正常标记未保留，应保留该限制而不补造。

## 下一步实验卡（每模型最多两张）

1. **K101A：现队列继续＋Kmax组件。** 现64target+1draft不动。独立draft组件用910371/a02r3n15/1DCU、c8、64GiB、15min（先确认不占541设备）；只按maxKs裁head/selector，保留8hidden与attention。测Ks1/2/7及混合Ks、完整query/RPC与子项；通过后才做真实AR/spec对照。完整wall无收益则停，不承诺wholequery7x。
2. **K101B：完成416→条件式shard-only Model。** 现910380/a14r2n08,n09/16卡完成M1/4/8/16/32/64 ABBA和独立显存模式。真实head＋synthetic hidden不是Model质量证明。通过后做新head/model对象闭包与loader切片，原四节点n08/n09/n10/n12上的32卡PP2C16只在资源释放并检查空闲后执行；所有owner实际constructor/loading/warm/max-context>4GiB。2.0508GiB节省、5.1191GiB tail空闲仍是投影。argmax API延后。
3. **G101A：B32 helper→一次Model M32。** 已冻结manifest31eb1：910376/a02r4n14/1DCU、c8、32GiB、600s，测B17/31/32完整helper。通过后单独闭合caller/Slots/CAPI/scheduler边界；n12/n13共16卡同驻留C32M16 raggedON对C32M32，真实EOS退槽、Bhist、质量及完整common window。不能只改guard。无Model净收益则保留M16，停M32。
4. **G101B：同驻留复用已有window hooks。** 源码确认varlen几何支持后，在G101A同驻留取一次短decode窗口、立即read/reset，分清receive/send、submit/wait、attention/FFN/control。hook不支持就跳过本诊断，不阻组件，不新增profiling框架。stage1约两倍elapsed包含等待，不能解释为两倍计算。
5. **D101A：attention内部分解。** 910374/a02r4n05/8卡、c64，沿用1065资源/EDF对象与既有attention fixture；同真实B4/1681token做有界OFF/ON/OFF，分解实际projection、RoPE/cache、attention core、Oproj、collective/wait。先检查已有接口；必要时仅当前private attention TU加少量marker。测完选择唯一明确热点。
6. **D101B：条件式唯一改动。** 根据D101A测得热点，进行一个attention或host/wait改动；先完整component配对，再同负载uninstrumented Model A/B，保持B32 compaction及B8/B16 fixed行为。没有明确热点/完整收益就停，不做HC/MoE大网格或重复confidence回放。

跨模型 **H101A** 优先级低于上述已预留资源：为CPU3491的单HOST raw AllReduce补最小driver，BF16/F32、229376B/1/16/64MiB、3warm+10样本、rank+1 out-of-place和136；记录enqueue/drain/event及实际选路。16卡n08/n09只能在416与后续head工作释放后串行使用，不抢占。整数collective校验不等于模型token一致门。

所有实验允许不同token；早期合理、后期benchmark无明显掉点，不恢复F64/bitidentity。每次保留失败、EOS/cap、完整wall与参与卡分母（含draft）。不读取raw nonce health、不碰910384或其他live；独立已授权工作无需等待本review。详细exit条件与精确资源、13份哈希见JSON。
