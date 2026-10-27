Astra review 116 — 本地源码与下一步投入审查

结论：继续 Kimi 2k 的实际 CPU 编译和完整 MMLU32 质量验证，性能方向只增加一项当前 C2/W8 的 T16 时间归因；推进 GLM 精确 M15 投影的整模型集成；Deep 先修编译上下文并完成八 owner 事务入口，再做原生前缀资格测试。当前没有三个模型新的共同达标结论。

输入 JSON SHA256 为 7eb74778f3380f44fe3629a7d9bd5e8f2b8b3409d0a4f07aa438fd080375eea1，8 项引用全部匹配；进一步核验 Deep manifest 的 51 个文件和 Kimi source-pins 的 76 个文件，均无差异。读取仓库 AGENTS、质量政策和 Astra 政策。仅本地读源码、写本报告；没有 Git、SSH、上传、编译或 GPU 作业。JSON 是完整卡片与字段记录。

本次仅消费两个已完成实验族：910379.474+910371.586 整模型一轮，以及 910376.758 组件一轮。CPU 修复、编译和本次审查不增加轮次。此次是可避免脚本错误触发的提前审查，不写成已满三轮。

已有收益与边界：

- Kimi GSM C2 的整体 decode 从 24.9257 到 25.7394 tok/s，+3.264%；逐用户中位从 13.1725 到 13.6945，只有 7/32 严格超过 15。C8 为 0/32。32 target owner、draft 均正常结束，post 空；仍是 native emitter 指标，首末 token 区间包含其他请求 prefill，并非 HTTP 服务。MMLU31 为 24 正确、24 自然 EOS、7 cap、23 同时正确且自然 EOS，不能称完整质量通过。
- GLM 原始 OFF/ON 时间重新计算四对倍率为 4.47525、4.45759、4.46431、4.47133，中位 4.46782x。原结果的 median_speedup=3.46782 实际是倍率减一，应描述 +346.782% 速率，不能描述 3.46782x。NRMSE 为 0，清理和退出通过。范围是构造 BF16 输入、训练权重 layer38、量化+投影+drain；只有 co14336，完全不构成整模型倍率。
- Deep r124 和 Kimi r123 的 CPU 静态契约是源码工程证据；Deep 没有原生执行，TP32 只有单 HOST TU 编译，均无新的 Model 性能或运行显存结论。

具体源码问题与处理：

1. Deep compile_entry_r124.py 从链接 argv 提取所有 HOST 编译参数，无法证明每个原对象的真实编译上下文。应从九份原对象编译命令恢复完整参数，仅替换源文件、输出、depfile 并前置私有 include。现有 MD 检查只排斥 src/*.hpp 同名外部文件，未要求必要头存在，也未覆盖根目录事务/adapter 头。需要按 TU 检查必需私有头与旧布局头排斥，不能只声称 exact binding。
2. Kimi CPU_build_r123.py 的 ABI layout 命令截取 -MD/-MMD 前缀，可能丢掉后续语义参数；必须从完整实际 argv 删除特定源文件/输出/依赖控制参数。infer_engine/allocate 当前复用 worker 的编译命令，应恢复原 TU 上下文或证明等价。此问题应在一次六 HOST 编译前解决，避免重演 TP32 metadata 失败。
3. Deep publish_prefix 目前立即本地发布，八 owner 同意并未实现。事务须绑定同一个 epoch/cohort/accepted vector，全部 seal 且 stream 完成后才允许发布；部分 rank 已发布后故障应整个 cohort fail-stop，不能声称可回滚。per-row generation/reset 不存在：第一轮原生 harness 可固定 cohort、病例间 whole reset；真实服务复用必须补齐行生命周期。
4. Deep 的 accepted partial seal 保留所有 43 层 next_state，且可能再次复制整套固定 state。配置物理行每行 6,103,040 字节 pending，partial 再加同等规模；ring COW 每层每触碰物理页 149,760 字节，两行共享一页。baseline、score/query/topk、驱动和 allocator 还另算，667kB 绝不是实际上界。需测 forward/seal/publish/cancel 的实际高水位。
5. Kimi 新 helper 名规避旧 inline 弱符号选择；16 physical/132 pages/8 pages per request 与 pool17 的映射源码未见明显越界。136 项静态检查不能证明真实 2k attention、replay、ABI、draft position2054 或 128MiB frame 的临时空间。仍按 W8、projected cache OFF 资格测试，不用静态 1.029GiB 余量代替运行 fit。
6. GLM r125 草稿的 exact geometry guard、dtype/owner 检查、独立 default-OFF handle 基本方向正确。co0..14336/步长2048 尚需所有八 rank 的训练 scale 数据覆盖；policy setter 自身没有 reset/idle/lease 约束，必须由 owner bridge 落实。31,125,504 字节私有 workspace/owner 还需和原缓存共同计账，验证 clear/reuse/destroy，不重复计入预留。

下一步卡片（每模型最多两张；原 owner 执行，root 记录采纳/拒绝/暂缓）：

| 卡片 | owner、资源与最小实验 | 首个结果预算与退出条件 |
|---|---|---|
| K116A | Kimi ctx owner：先修完整编译参数，实际编译六 HOST TU 与 C++ 契约；随后现有 TP16 PP2 的 32 target+1 draft 单次加载，固定完整 MMLU32 OFF/ON、cap1024、不截题干，先8题再其余24 | CPU 目标30分钟；已知加载约28.5分钟。所有长 prefill/verify/replay、draft 查询边界检查4GiB floor；生命周期/显存失败或明显质量回归停止本路线。采样 floor 不是峰值证据 |
| K116B | Kimi profile owner：只做当前 C2/W8、W2 路径 T16 时间归因。两暖场、四组平衡 observer OFF/ON 配对真实 C2 cohort，分 draft/verify/replay/commit/emit、prefill interference、T16 MoE；尽量复用一次已有合格加载 | 加载后额外测量≤15分钟。现有739只有T8/T64。固定 token 数下从13.694到>15至少需减少>8.70%首末时间；找实际可移除部分，不能以中位推断32人全过。observer 扰动超过配对波动则缩减 trace，保留OFF时间 |
| G116A | GLM owner：补 r125 bridge、完整CPU闭合，再现有TP8同加载OFF/ON；先八 offset 与清理复用检查，两暖场后四对真实15请求ABBA cohort，保留T16未改变控制 | CPU目标30分钟；排队加载未知，测量≤20分钟。unsupported shape 命中、counter/lease/清理错误、质量或wall回归即停止。收益只报实际 Model wall和逐用户时间；不预测4.47x |
| D116A | Deep owner：九replacement+一adapter完整HOST上下文、实际事务C++测试，八owner协调/worker/CAPI入口与明确reset契约 | CPU首个结果目标45分钟；整合时间未知。拒绝旧布局对象混用、私有头缺失；常规include/searchpath修复在已有授权内完成 |
| D116B | Deep native owner：D116A后固定cohort TP8原生前缀测试，覆盖accepted1/2混合、2/3/126/127/128边界、EOS/cap/cancel/stale/reset、发布前故障、随后有效继续，检查全部43层与compressed/indexer有效性 | 排队加载未知，测量≤20分钟。不加入完整MTP head；任意跨rank不一致、无效字节可读、行复用或floor错误即停止。本轮只资格验证，不声称加速 |

选择 K116B 而不是立即改 W16：虽然 C2×16=32≤64，当前 payload、state、Python native frontend 的 width 仍≤8，draft query K≤7，训练 block8。必须整套机制与接受率真实证明，不能只改上限。prefillmicro 调度则等 first-last 中实际 prefill 干扰被量化后决定，防止优化 TTFT 却误称 decode 改善。

TP32 只运行已经准备的 r120d metadata-only verifier：1 CPU、2GiB、3分钟，无编译、重链接、dlopen 或权重。保留441失败记录与已生成DSO，完整参数 fresh-MD 一致后继续组件工作；重复16行/8distinct的teacher输入不能当真实用户流量，也不能推断93层/head/H3 fit。

迭代成本：Kimi target实际72m41s，draft71m18s，其中加载1710.529秒；更细准备/排队分项现有证据不足。GLM758计算作业39秒、keeper53秒。CPU438加载路径、440 strict flag、441 filtered flags 都是可避免工程错误，零性能轮次。已编译不变 DSO 的 metadata/searchpath 应直接修验证，不能重编或新占GPU。下一次审查检查本次卡片实际落地与结果，不重复提案。

这些内部 source 开关和 root-reviewed 参数是执行约束，不意味着用户缺少授权。root 在现有授权内把具体文件、资源和退出条件发给原 owner 即可。
