# GLM attention detail: source-only draft

本目录是 actual CPU143 / GPU146 来源链上的独立观测源码草稿。`source_only=true`、`GPU_GO=false`、`target_compile=false`。未编译、未加载 Native、未运行设备或生成 Model caller；不是新的性能或源资格证据。

## 父闭包与修改边界

`PARENTS.json` 冻结三个 CPP 与五个逐字节不变头文件的路径、SHA。NoV 来自 `indexer-selector-timing-CPU-v1`；prepare/output 来自 `GLM952-global-TP16-component-r470/full-native-source-v1`。prepare 对应 CPU1119 plan unit4 并被143复用；output 是 r470 原复用对象。actual143 新编 NoV object SHA 为 `bcfdef3969f2c20ce4c45baa6ad6f9dc3832f689af9242daf9d1fb604b152ddf`，库 SHA 为 `a98a093aab52f2e4d85a2d11e3407a033909daf9e247c643bffce6fc8b64b1bb`。146 `native_arm.py` SHA 为 `cc82347933bd1e2edf10268de18c8fb75e9a175626eea06ff829756fbcb454ea`，精确检查143回执和该库；邻近 `CPU_BINDING.json` 只是1119祖先。

只修改 NoV、prepare、output 三个 HOST TU，增加本地 `attention_detail_v1.hpp`。原 `glm_sparse_observer_r131.hpp` bytes、State、Mark、TLS、CABI、512 event pool、原三帧选择及所有 Work 析构保持不变。没有新增同步、提前同步、retire、GPU TU、数学、布局、通信参数或分支策略。原 marker 保留原名称和位置。缓存 provider 调用只执行一次，保存原 bool 后进入同一 fallback 条件。

## 独立启用与观测 schema

默认关闭：只有环境 `GLM_ATTENTION_DETAIL_V1` 精确等于字符串 `1` 才可能记录新标记。缺失、空、`0`、其他字符串均关闭，不抛异常。先检查原 active/enabled/detail 状态、实际 Runtime 注册身份、原 C1 选择表、真实单行 T1 continuation、原生命周期 frame 1..3、已选 full layer0/shared layer3，再读取环境。未选择 C1、prefill、多行或其他层没有新增 mark。未来 caller 必须冻结该环境和新事件预算；旧 caller 仍受其278断言约束，不可直接启用。

复用 `detail.layer.<layer>.attnd1.<site>.*` marker，底层仍是原 `marks[{name,owner_host_s,gpu_from_first_s}]`，不新增持久状态。begin/return 表示 host call 与同 stream event 包络，包含 enqueue gap、依赖和原等待；不是独占 kernel 时间。原 drain 后的 Query/ElapsedTime 读取机制不变。

投影 tuple: `M/N/K/row_offset/column_offset/input_BF16/contiguous_checked/canonical_stride_inferred_<K>_1/raw_stride_UNKNOWN/kernel_provider_UNKNOWN`。M 是实际输入行数；dtype、连续性来自原输入校验；output 原连续三维输入随后作原二维 view。`[K,1]` 只是 canonical contiguous stride 推断，**没有读取 raw stride，实际 raw stride UNKNOWN**。`dispatch.cached_r809` 或 `dispatch.fallback_Fp8BlockLinear` 仅记录原 bool 路径，**具体 kernel provider UNKNOWN**，不能据名字声称实际 provider。

## 窗口与未拆残余

- NoV: forward_latent、fresh expand、selection D2H、CPU remap、ids allocation/hold/H2D（内嵌纯 H2D 调用窗）、history selected-value、key gather、条件性 value gather、两条互斥 attention 路径、result copy、output caller、原 stage-finish/stream-wait。
- prepare: 保留原 quant/projection markers，补 tuple、实际 dispatch、原 success_fence 两侧；latent-only 与 expand_cached 原 syncStream 两侧。
- output: tuple、quant、scale 初始化/scatter、projection/dispatch、copy、allreduce、原 success_fence。

history selected-value 的 begin/return 包含复用头内 latent/rotary gather、原 expansion 和相关分配；内部 gather 未新增独立 marker。fresh/history 由外层嵌套窗识别，同名 projection 按 occurrence 顺序关联。history lambda 在原 packed 条件内记录 `history.dispatch.packed`，在原 standard 返回前记录 `history.dispatch.standard`，两者互斥；与 complete-prefix metadata 并列记录 `v398.enabled/disabled`。history return 标记仍在原 expanded 赋值之后，表示 host 调用包络，不是函数内部 kernel 结束点。`gather.value_conditional` 包含原 `if(!v398)`，条件不执行时只是空分支包络，不应计作实际 gather。缺失分支表示未执行或无有效记录，不填零耗时。

Work 析构和异常 drain 内没有插桩。prepare/output 原 retire、NoV Work 的原析构 wait 可出现在内层 return 至外层 caller return 间；这部分是 **UNKNOWN residual gap**，不能归为 kernel/通信，也不为了补窗改变生命周期。error/unwind 的不完整帧不能当成功成本证据。

## 严格静态预算与 verifier

`EVENT_BUDGET.json` 给出每个新增调用位置的最大执行次数和贡献，sum 为 **162 新 events + 原278上界 = 440 <= 512**。这是保守静态上界，非实际计数；同一层互斥 certified/selected、history 两个返回分支及 packed/standard 两个 dispatch 分支均计入上界。新增两处 dispatch 与一处 v398 metadata 各按两层计算，共增6个 event。

调用次数依据：每层一次 latent prepare 中 qa/qb/ka 三次；full indexer 最多 wk/wq 两次；fresh kb 一次；C1选中逻辑行数不超过 topk2048，小于历史 block8192，因此 history kb 最多一次。full最多7、shared最多5，预算保守按两层各7计算。每层仅一个 C1请求行；expand最多fresh+history各一次。support 原件路径/SHA 和相关源调用结构在预算文件中固定。

`verify_source.py` 只读本地文件并向 stdout 打印 JSON：

1. 核新 MF、所有父源及五个不变头；逆转 `SOURCE_EDITS.json` 的精确替换应恢复原 LF 与行尾空格/tab 规范化全文（SOURCE_EDITS.old 与父源 bytes/pins 不变）。
2. 独立删除明确标记的 observation additions、还原两处“保存一次原cached返回值”的改写后，C++ token 序列必须完全等于父源，包括数学、条件、同步/调度调用和参数。检查原同步/dispatch调用次数与 Work 析构全文不变。
3. 从每项 edit 计算新增 mark/tuple/dispatch 调用数，乘以绑定父源推导的执行次数，再验证440预算。检查 helper 的完整gate、独立默认OFF和无新同步/事件分配/CABI。

该检查不等于 C++ 编译、ODR/ELF、设备执行或运行时等价证明。观测本身有 host/event 开销，不能把开启观测的 wall 当作优化收益。

## 未来独立 CPU 门槛

详见 `FUTURE_BUILD_CONTRACT.json`：以 actual143 为父，**3 HOST 新编、29对象复用、0 GPU TU**，不使用 projection-single-drain/新 selected-value 优化包。沿 NoV 的143 `NOV.json`/`build_once.py`、prepare 的1119 `BUILD_PLAN.units[4]`、output 的 r470 `ARGV_ANCESTRY` 精确 recipe。新 TU 的 include 搜索必须落到本包字节不变 observer 头；fresh `-M` 与实际 `-MD` 的路径和逐文件 SHA 对齐，校验预算 support 源与目标依赖相同，否则预算不具备目标资格。

保留32对象 link顺序、280强导出、71 provider、原NEEDED/ELF约束、observer/selected-owner TLS唯一性和强timer getter来源；重编三个 TU、其余29个对象须从actual143闭包逐一核SHA，不能凭当前源码目录替代原对象证据。原 `modern_glm_pipeline_last_profile(void)` 返回同 owner thread 的 TLS double[6]，顺序不变，forward返回后立即复制，下一forward前完成；不改CABI。

未来 caller/read verifier 需同时固定新env和derived440预算，保留512池及原3帧。此目录不提供 caller，不授权编译、训练、GPU运行或重开负族。
