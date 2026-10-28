# TP32 I96 配对组件候选 r163

状态：本地 Python AST、物理几何和静态分支 contract 通过；尚未实际 HIP/Native 编译，尚无 GPU 候选结果。默认关闭。这里只批准 CPU 编译准备，不自动批准 GPU 或整模型加载。

最快区分的问题：纯 TP32 已有 generic I96 可运行证据，但当前服务的 TT2/Tile4 W13 与 pairW2 专用分支只覆盖 I192。将相同解码、FP32 乘加/归约、SiLU 与 BF16 输出参数化到 I96，验证其能否显著降低 TP32 单 MoE 层时长。保持原已资格 provider.o / packed.o 及其 packRoutes/buildJobs kernel stub 不变，避免重复已知 LLVM 原子 warp-aggregation 编译失败；严格链接失败即停止，不转入 GPU。

## 已有证据与边界

- 仓库 `results/bw1000/2026-10-27/kimi764-tp32-i96-moe-component.json`：实际 910375.764、32 ranks、TP32PP1 generic I96，单 checkpoint 第 1 层。T8/T16/T64 的 max-owner 时长中位为 1.418964 / 2.077841 / 5.751677 ms；全部 32 owners 正常退出。历史输入仅 8 个不同 token 行，T64 重复，不冒充当前服务流量。
- `results/bw1000/2026-10-27/kimi-tp32-constructor767-and-fixture466.json`：910375.767 实际 TP32 constructor 正常。只有单 KDA、单 MLA、head、93 层 cache 及通信元数据构造，没有全模型权重/forward。无需重复 constructor 或 generic-only 测试。
- `code/bw1000/resume-20261026/TP32-actual-constructor-qualified-r120k.md`：实测 GPU 可见 63.984375 GiB；全 93 层 WORKSPACE_OFF 静态估计 58.906441 GiB。保留 4 GiB 空闲后，其他所有未计峰值仅有 1.077934 GiB，且非重叠未证实；WORKSPACE_ON 静态估计已超此预算。整模型 fit 维持 UNKNOWN_STOP。
- 真实 KDA heads=96，TP32 每 rank=3；vocab=163840，每 rank=5120。routed intermediate=3072，TP32 I=96、packed W2 stride=48、scale stride=3，几何可整除。92×I96 与 46×I192 的专家项接近，replicated 模块覆盖层数增加，因此不能称单卡权重减半。
- 历史 TP16PP2 对 TP32 的跨运行组件数值不能证明因果收益，不能除以 2、乘以 93 或外推每用户/总 tok/s。减少 PP 串行路径只是待证假设。

## 源码与 CPU 入口

包部署到 `/refactor/incoming/Kimi-TP32-I96-template-component-r163` 后，由根协调者在既有 CPU keeper 中运行 `bash CPU_inside_r163.sh`。不使用本地 SSH/Slurm，不启动新 keeper。脚本只编译/链接，不导入 torch、不 dlopen Native、不触 GPU。

父 receipt：`/refactor/candidates/Kimi-MoE-phase-component-r155/CPU-build.json`，SHA256 `c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd`；父 Native SHA256 `d559f6a60081b50aa43dcb5a442119c38d62cc757a5c3d35619c599dcef2dc2b`。精确复用父 compile/link command、所有原 object SHA 和真实 .d header SHA，仅替换 HOST MoE object、追加 I96 HIP object、编译单层 HOST adapter。`--no-undefined` 和 `ldd -r` 必须通过，包括外部 packRoutes/buildJobs launch stub。

输出目录 `/refactor/candidates/Kimi-TP32-I96-template-r163` 必须不存在；失败保留证据，不原地覆盖。CPU 脚本内部总 timeout=900 s，所选 CPU keeper 剩余时间应超过 16 分钟。实际 compile/link 未完成前不发布候选 GPU 准入结论。

## 条件 GPU 实验（根协调者另行授权）

只允许根协调者持有的 910380、4 节点 a14r2n08/n09/n10/n12 各 8 卡，须先普通发布源码/认领，完成 CPU closure、fresh 空闲/资源核验，并准备 independent controller/post-cleanup。这里不提供 Slurm 提交器，也不触 910384/n17/n18。

同一新 Native、同一实 checkpoint 第 1 层只加载一次，A=SDK generic I96，B=原 TT2/Tile4 I96 + pairW2；不混入新 directdecode/wavepartial。4 个工作负载为历史 T8/T16/T64 +固定 seed 的同 RMS Gaussian64 控制。每 workload 4 次暖调用后两轮 ABBA，每 owner 共 48 arms。所有 32 ranks 的 scalar allreduce barrier 在计时之外。候选 hit count 必须每 B=1、A=0，旧 I192/provider counters 不变；finite、输入/尾 canary 只读、组件 NRMSE<=0.01 为诊断边界，不作 token 完全匹配门槛。不写原始激活或原始输出。

仅增加单层固定 scratch=3,878,656 B；不改 C++ class layout。GPU 内部 hard timeout=660 s。运行 runner 须提供 `MANIFEST_R163` 路径，其 JSON 字段包括 `GPU_authorized:true`、`source_published:true`、`publication_branch:"bw1000-experiments"`、`parent_job:"910380"`、上述 `expected_nodes`、本包 `source_manifest_SHA256`、CPU receipt path/SHA、fresh master_addr/master_port。runner 校验真实 step/node/card/CPU affinity、源码与 Native SHA、dladdr 后才加载单层。

`analyze_r163.py node0.json node1.json node2.json node3.json NEW-result.json` 只生成新文件。所有工作负载两轮 ABBA 均 >=10% MoE 收益、OFF endpoint 漂移<=3% 才 eligible_for_later_design。组件成功仍不自动放行全模型：必须另行解决上述峰值显存证据，模型/质量/逐用户 decode/总吞吐均 UNKNOWN。即使适合继续，也只在未来必要模型加载中做同负载 C8 ABBA + C1 控制。
