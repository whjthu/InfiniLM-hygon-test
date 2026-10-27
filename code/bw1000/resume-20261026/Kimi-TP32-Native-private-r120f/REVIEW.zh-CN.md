# TP32 Native r120f 源码交付

这是可评审的独立 HOST 实现源码，尚未编译、加载或做 GPU 验证。Native305726、运行中的 384、r120、共享 SDK/Runtime 全部保持原样。本包没有 Slurm/GPU launcher。

生产路径只替换原 Native 的五个 HOST object：pair_probe、worker、delta、head、policy。MoE、I96 generic SDK、旧 provider、MLA、cache allocator、CommunicationGroup 等对象按原 SHA 复用。可选构造诊断再增加一个独立 HOST wrapper object，不链接进生产 Native。

## 已实现的兼容改动

- 新 `INFINILM_KIMI_TP32_R120F=1` 明确启用 TP32PP1/cap8；所有 I192、WORKSPACE、Math 可选分支默认关闭。构造前验证全局 KDA/MLA heads96、局部3heads、I96、D128/MLA192、真实 physical16/state17/pages68。TP16 路径仍沿用其原开关。
- batch ABI 的 flags3、epoch、retained prefix、row swap、EOS 和 owner fence 保留。TP32 将传给 worker 的 grouped-prefill bit 清除，T<=512 只走原 generic 路径；verify 仍总T<=64。不会把 I192 provider 用在 I96 指针上。
- batch/state-only capture 从实际 gate shape 检查 H3 或 H6，推导 P=H*128、QKV=3P；replay 使用已保留且已校验的同层几何。保留旧 owner/epoch/pool-pointer 和 drain 检查。H6-only serial/rowAR/prefill512 可选路径由 TP32 环境契约关闭。
- shard-only head 支持 TP16/32，TP32 实体为5120×7168 BF16/73,400,320字节；CPU first/mid/last 是0/2560/5119，expected digest 分区数与实际head分区数必须匹配。保持全量 BF16 logits allgather 和 fence。
- 私有 Python runtime 改为4TPnodes/PP1、sampling node0、全93层归属校验、32 target+1 draft 的卡数计量。它的 full working set gate 默认拒绝；本轮不能运行该 entry 或把它当作完整 benchmark frontend。

`HOST-changes-r120f.patch` 给出实际 C++ 变化。worker/delta 本体字节不变，因包含改变的 state header 必须一起重编。头类名/现有 CAPI 没改，必须通过 CPU builder 的 ODR scan 验证没有保留旧 inline/head 定义的对象。

## 内存证据与限制

764 已证明32 real owners 的 I96 MoE 正常执行，每 owner 实际 state_dict 逻辑参数615,197,184字节。收到的 r9320 汇总没有 CUDA free/total 或峰值；不能把逻辑参数冒充 allocator 占用。

重算的全93参数62,163,168,060字节，cache1,086,011,136字节，WORKSPACE_OFF私有scratch1,130,496字节，静态合计63,250,309,692字节=58.906441GiB。名义64GiB扣4GiB floor只剩1,174,199,748字节，尚未覆盖真实SKU/driver/CCL/allocator/算子与生命周期峰值。WORKSPACE_ON静态60.546068GiB在未加这些开销前已不满足4GiB余量。详细分类见 budget-r120f.json。

不把764单层时延乘93，不推整模型 TPS、最佳机器数或15tok/s上限；H3 KDA/MLA forward/replay 与 wholeModel 质量仍未验证。

已有743的TP16同历史layer1输入，T8 first-OFF arm median1.3848615ms，四对OFF median1.373704ms、pair-ON median1.30020475ms；764的TP32 generic T8是1.418964ms。T64对应743 OFF6.2928325ms、pair-ON5.6263745ms，764 TP32 generic5.751677ms。两者都计时完整MoE+outerdrain、报告max-owner量级，但属于不同轮次、16/32 owners、不同provider/workspace、不同预热/重复/交错顺序；743有显式owner affinity核验，764源码没有同样的显式pin核验证据。这不是严格A/B，不能预设TP32 MoE更快。纯TP32后续主要待证明减少PP和attention成本是否弥补genericI96效率。TP16×EP2保留I192并按专家分片仅作后续独立任务方向，本包没有EP实现或新候选扩散。

## 明确的下一动作

1. root 评审、发布源码后，仅运行 masked CPU_inside_r120f.sh。它保留真实完整编译命令前缀，编译5个 HOST TU、运行纯CPU C++尺寸/row生命周期契约，核对 actual/fresh MD 完整集合、原对象 SHA、ODR、strict link/nm/ldd。任何失败保留 false receipt，禁止降门槛。
2. CPU成功后，可另行用同样 masked 环境运行 CPU_build_construct_r120f.py，编译1个无权重构造 wrapper。此步骤也不 dlopen/创建 Runtime。
3. 再单独审阅并发布未来32-owner构造控制器，使用既有375四节点真实32卡。`construct_TP32_r120f.cpp` 只构造一个H3 KDA、一个H3 MLA、一个head和原93层cache，不构造93层参数、不打开checkpoint、不执行模型forward。使用独立已初始化的BF16 rank-tag输运 fixture验证32×5120 allgather，明确不是用户激活。记录各 owner 的free/total/used及阶段差额，初始free>7GiB、各阶段free>4GiB；正常关闭全部32 owners，Slurm终态后postKFD。
4. 构造成功只关闭尺寸/cache/head-collective与部分driver基线缺口；后续仍需有权重的H3 KDA/MLA及partial replay小组件，保持原GPU math0，确认数值基本合理/finite和状态生命周期。若原算子不支持，先记录确切失败再提最小GPU TU，不能直接换旧H6 kernel或静默放宽。
5. 只有完整的实际SKU、driver/CCL/allocator及同时存活临时量上界证明严格>4GiB余量后，才生成新的全93 constructor/load GO。当前完整Model gate为false，不能因为subset constructor通过而自动改true。

本轮仅本地源码/JSON/AST/契约门禁检查，0 SSH、Slurm、上传、C++编译、dlopen、权重读取、GPU、Git。
