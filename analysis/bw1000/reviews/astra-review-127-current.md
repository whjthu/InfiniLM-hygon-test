# Astra 127 当前审阅（2026-10-29，提前触发）

结论：保留 Kimi 真实 C1 pilot 成果，停止为 registry 再加载整模；TP32 容量仍为 **UNKNOWN_STOP**。下一步 Kimi 先做专用 persistent MLA cache exact + 同 owner 共享 I96 scratch 的组合源码设计。Deep 把完整数据检查前置并与 CPU 共用同一入口后，只运行一次真实 C1；GLM 已授权的唯一 keeper 独立继续。

本轮提前审阅原因是 Deep 1187 在加载后触发复制的旧数据 guard，重复 515 类可在 CPU 发现的失败。本审阅只读本地证据、写 V 报告；未 SSH/Git/启 GPU，未修改 repo/status/ledger。

## 实际收益与时间去向

- **Kimi 456：4 个 C1 pilot 的 user 为 16.26、19.45、18.85、20.37 tok/s。** 这是直接单用户速度证据，已经超过 15 tok/s；范围仅这四个 pilot。两组 C8 的 user p50 为 5.74/5.01，aggregate decode 为 37.20/33.80。20 次执行全部正确、EOS、0 cap，只有 16 个 unique。固定 cache ON / W13 OFF / pair2；没有匹配旧 C1 A/B，因此不能称本轮新算法收益，也不是在线服务或完整 heldout 验证。
- 每节点加载 1213–1581s；Native 909–1235s，get_tensor 286–361s。每 owner 两次 state_dict 合计仅 0.35–1.48s，约不超过加载的 0.13%；继续优化这一项没有实际价值。405–428GB read、约 900 万 major fault 与分页/读取有关，但没有足够分段证据将 Native 壁钟归因于纯 IO 或 H2D。保留当前缓存实现，加载归因随下次必要 Model 运行采集，不为此单独 reload。
- **1189/1191 的最薄真实 helper probe 有效**：两个 fresh 单 owner、4 个 storage、5376 aliases、dladdr、正常退出和 post empty；实际 regular/exact 构建增量 744/618MiB，差 126MiB，比设计少 3.4375MiB，已闭合原分配差目标。不要重复 probe 来修正文案。
- **Deep 1187 的 8m29 没有产生任何 C1 trial。** 它是可避免的工程预检漏覆盖，不是性能 negative。CPU_static 只查四个 ID 存在，未执行 run 中旧 fresh64 全量 guard；tokenized SHA 通过不能覆盖 dataset namespace 与 admission 契约。保留 FAILED、engine join true、post1193 empty，不自动重提。
- GLM 524/528 CPU 已通过，31a3/d078 已发布，唯一 keeper 已授权。实际 GPU 状态由 root 核实；本审阅不推定开始或完成，也不阻塞它。

## 显存解释与容量账

Runtime_background_delta=0 只是 after_Runtime − before_Runtime。两臂 before 为 322MiB，after_Runtime_destroy 为 468MiB，**共同残差 146MiB 的归属 UNKNOWN**。regular/exact 在 Runtime destroy 下降 598/472MiB；这是阶段差，不是直接 pool reserved 测量。KFD 在进程退出后为空，不能证明 Runtime 已恢复全部显存；同样不能据此判 leak。不得把共同残差乘 92，也不得与 767 baseline 不加辨别地累加。

| 每 owner 项目 | bytes / 解释 |
|---|---:|
| 实际可见物理内存 | 68,702,699,520 |
| 保持的 free floor | 4,294,967,296 |
| 旧逻辑 static | 63,250,309,692 = 58.906441GiB |
| MLA cache class 额外开销 | 648,019,968 |
| I96 scratch 相对旧 OFF 的 logical 净增加 | 355,705,856 |
| 新 scratch class 额外开销 | 29,039,616 |
| 加以上项后，opaque 开销前剩余 | 124,657,092 = 118.88MiB |
| 仅 MLA exact 后剩余 | 772,677,060 = 736.88MiB |
| 再以单 owner 一个 4MiB scratch 替代 92 个的条件性剩余 | 1,154,358,724 = 1100.88MiB |

**58.906GiB 不能减去 11.629GiB packed padding**：旧账本从未包含该 padding。126MiB 单层实测不能冒充 92 层实际峰值；exact 也不保证底层物理 malloc 每字节精确。767 曾采样到约 848MiB baseline，但持久性及重叠未闭；即使组合方案成功，operator、load/finish、KDA/cache、参数唯一 storage 与通信高水位仍需核清。当前没有 full93 Model GO。

## 下一张卡（每模型不超过两张）

### K127A — Kimi 容量组合设计，优先 CPU

先改 Kimi 专用 persistent MLA KV bank 的分配：24 个 bank 走 exact，保持原 zero-fill、shape/stride/dtype、owner Runtime、cache vector 与寿命，不改全局 allocator，也不能用 bool=true 冒充 exact。入口见 `Kimi-TP32-Native-private-r120f/kimi_shard_model_r9183.cpp reset_cache`；清零参考 `Kimi-PP2-cap32-budget-private-r9201/source-proof-r9201.json kv_cpp:204–211`。

同时检查 `Kimi-TP32-I96-template-component-r163/kimi_moe_I96_r163.cpp:299–321` 的 model-owned workspace。共享必须是同 owner/generation 的真实 scratch lease，源码证明层间不重入、异步消费者结束后才复用，且 replay、并发调用和不同 stream 不冲突；不能只因 host 顺序调用就假定设备生命周期结束。若寿命不成立，保留独立 workspace，不能强行共享。

cache 设计可去掉 648,019,968B；共享设计另去掉 91×4MiB=381,681,664B。两者只是容量设计，不是性能收益。CPU 首结论预计 15–30min。合并设计和字节账核清后，再决定一次生产共享 helper 的薄分配/生命周期验证是否有价值（约 2–5min GPU 执行）；当前不直接启动 probe/full93，不重测 constructor767。warm/perf 不适用于本轮 CPU 设计；decode wall 份额 UNKNOWN。opaque 峰值未闭，继续 UNKNOWN_STOP。

### K127B — 保留可用 C1 基线，暂停追加整模加载

456 的固定路线作为真实 C1/C8 基线。下一次必要 Model 运行应直接回答用户收益：同题同预算，分别记录 user、aggregate、task、TTFT/ITL，并保证基础正确性、EOS/cap 无明显退化；不要求 exact token。不为 registry、约 1% 的无大 wall 依据候选额外花 20–26min 加载。容量闭合或有可观 wall 份额候选后再启；暖机与固定 case 沿既有卡，不另做扫描。现阶段零 GPU 新负担；未来首结果至少为现有加载加试验时长。

### D127A — 修复同入口数据契约，再唯一 C1

把 pure prepare_cases/load_case_plan 同时供 CPU 与 run 调用，且在 Runtime/Model 创建前执行。锁实际 prepared/dataset/tokenizer SHA；检查 accepted+skipped 的完整分区，以及 selected4 唯一、未拒绝、ID/原题/label、tokenizer/template、cap/no_crop 一致。按实际 9278/159 数据命名空间修复，不能盲删所有 guard，也不能把旧 fresh64 全量 equality 搬到新任务。

CPU 对实际文件完整执行此函数，再验证错误 namespace/缺 selected ID 会拒绝即可；不加通用框架。原失败留存。root 已知终态和 post empty、同入口 CPU 通过后，唯一新 C1 作业完成原四题与 owner/event 来源。CPU 首结果约 5–15min，GPU 约 8–15min 起，取决加载与输出。没有任何 C1 wall 份额可用于优化推断；不要重复已负的 Deep 调度实验。未知 GPU 状态不重放。

### G127A — GLM 继续已有唯一 C1 keeper

沿 524/528 已合格源码执行真实 T1 decode observer，补齐 767 的 T16/T15 或 B1 prefix 没有回答的问题。root 核一次实际状态；若正在运行就等现有任务，不再另起。质量沿原卡，以基础正确/EOS/cap 为准；保留暖机和固定 case，不加 exact-token 门。用真实 T1 子区间 wall 份额决定下一算子，不能把 aggregate/C 当 user 速度。不再扫 GLM chunk。若尚未开始，首结果估 25–35min 含历史加载量级；若已开始按现有 keeper 等待。

## 硬件与收益边界

不虚构物理峰值，W13 scalarF32FP4 不套 BF16 dense 峰值。组件倍率 s 的整模理想上界为 1/((1−f)+f/s)，f 必须取实际 Model 可优化 wall 份额；当前未知便不报整模增益。TP32 与 TP16PP2 的通信、owner、cache 和 batch 都不同，组件倍率不能直接搬到 C1/C8。逻辑少读/少分配不等于实测 HBM/cache 因果。FP4direct、W2quad4、专用 W13 Model、GLM chunk、Deep 调度均维持此前 STOP/PARK。

证据：`kimi456-registry-loading-C1-C8-pilot.json`；`runtime-results-root-r9512-a.json`；`Deep-failure-root-r9513-a.json`；`Kimi-I96-storage-1189-1191-compact-r168b.json`（SHA 724c8c2b…）；`TP32-fit-CPU-allocation-closure-r163b.json`；Deep r168 run/CPU_static 源码；126 审阅与 implementation。完整路径/数值见同名 JSON。

