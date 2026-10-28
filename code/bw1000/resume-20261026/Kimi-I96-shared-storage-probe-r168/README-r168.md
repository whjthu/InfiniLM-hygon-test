# 共享packed构造helper分配probe r168

真实MoE/Experts/MLP构造需要非空TP communicator，单GPU不能伪装TP32。按Astra126复核，本包把原`register_mxfp4_experts`的四bank分配、5376个narrow/squeeze、Weight构造及绑定顺序提取成同TU共享helper。原Experts构造保留全部comm检查，继续调用该helper；probe入口直接调用同一个helper并通过ModelWeights真实bind_weight绑定leaf。无假RankInfo、communicator、checkpoint payload、forward、cache或MoE scratch；测的是实际生产共享helper分配，不能称完整MoE构造或TP32执行。

regular/exact两份Native源仅四处allocatorAPI不同；CPU逐字节反向映射固定helper与原body等价。exact语义对应实际CPU522/r167 Native eddc95e620da4a3de06f03802ab8ba7c1bcc9cc4a27fd963b9c175dc3262b6d6，production r167产物不覆盖。每份只编一个HOST TU后链接，复用517其余33object，无HIP重编。

部署目录 `/refactor/incoming/Kimi-I96-shared-storage-probe-r168`，CPU入口 `bash CPU_inside_r168.sh`，780秒硬界；新输出 `/refactor/candidates/Kimi-I96-shared-storage-probe-r168/CPU-build.json`。本包不提交GPU。root后续在parent374/n01单卡fresh空闲下，以两个独立短step/新进程顺序调用 `python3 -B run_probe_r168.py regular|exact <本包实际CPUreceiptSHA> <publicationCommit>`；每进程只加载对应Native一次。需沿用实际CPUreceipt的LD_LIBRARY_PATH与原DTK环境，且只暴露一张GPU。

dladdr+SHA绑定真实probe入口/Native/source。实际记录beforeRuntime、afterRuntime、afterbanksConstruct、afterbanksDestroy、afterRuntimeDestroy的cudaMemGetInfo free/used，并逐bank校验shape/stride/offset及5376leafalias。Native pool reserved无公开统计API，明确UNKNOWN；不用Torch reserved冒充。保持每阶段4GiB floor，构造前还保留1GiB已知bank预算，无trim或allocator改动。

预期单层regular-exact增量差135,725,056B；设备实际页粒度/基线差额应与测值一起解释。92层12,486,705,152B（11.629GiB）只可标设计外推；这修正旧逻辑账本未计padding，绝不是58.906−11.629。全Modelpeak/fit和性能均未放行。root负责fresh/post KFD与两个step完整退出；本包不触其他作业。
