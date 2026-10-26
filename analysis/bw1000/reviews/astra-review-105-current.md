# Astra review 105

Kimi projected cache stays defaultOFF; prioritize nativeT64 adaptivewidth and real targetMoE evidence. Accept Deep sink as prefill opt-in with no decode gain. Repair GLM component lifetime then qualify once and promptly validate Model. No reviewed model reaches >15 per-user tok/s.

本次仅消费三组完整实验：`910375.731+910381.491+910371.571`, `910374.1079`, `910374.1083`。2 组 Model 矩阵、1 组组件矩阵；失败、CPU 准备、单臂和单题均不另计。三份 canonical 与两份 producer 的指定 SHA256 全部匹配。

Kimi 全 13 case 正常结束、audit0、post empty。以下吞吐为 tok/s，65 卡含 draft；C16 是两组16请求汇总。

| 并发/策略 | decode | 正确且自然 EOS goodput | 正确/EOS | 用户中位/最大 |
|---|---:|---:|---|---|
| C16 AR | 27.009 | 23.259 | 30/32; 31/32 | 2.640/3.550 |
| C16 state OFF | 41.016 | 38.099 | 32/32; 31/32 | 3.423/5.066 |
| C16 state ON | 42.248 | 39.262 | 31/32; 31/32 | 3.436/5.155 |
| C32 AR | 40.381 | 40.381 | 32/32; 32/32 | 1.713/2.319 |
| C32 state OFF | 43.679 | 40.554 | 32/32; 31/32 | 1.762/2.621 |
| C32 state ON | 44.331 | 41.157 | 32/32; 31/32 | 1.804/2.657 |

Cache 增益仅约1.5%–3%；C16 的1097题 OFF达到cap但答对、ON达到cap且未答，不能称质量无降。所有用户均未超过15 tok/s，满足正确且自然EOS的15 tok/s goodput为0。verify约占cyclewall的90%，但包含等待，且各时间项重叠，不能当纯GPU计算或相加。

Deep1079 是真实映射、合成RMS/RoPE输入的单卡CSA组件：四对 wall 46.258→38.978 ms（−15.74%），maxabs0。Deep1083 的8卡完整Model PF四对 10.215→8.891 s，输入吞吐+14.89%；B32含PF输出17.188→18.029（+4.89%），decode33.437→33.392（−0.13%，无收益）。配对32题两臂均正确且自然EOS；额外B16 ON-only后16题15/16单列。最大用户约2.09 tok/s，目标仍未达成。独立每owner normal标记未保留，资格依赖engine destroy/join、8owner身份及外部postKFD，不补造标记。

**K105A · kimi_continue_92 · STAGE_NATIVE_T64_ADAPTIVE_MODEL_THEN_ONE_PAIRED_MATRIX**

Keep projected cache defaultOFF. Compose immutable r9224 adaptive frontend with already qualified native b4f; width includes anchor, active32 W2 ->16 W4 ->8 W8, sumwidth<=64, perrequest remaining/context limits, exact once commit and complete partialreplay before newepoch. CPU8socket15cohort/86query/260retires is protocol evidence only, not GPU math. Freeze source and resource/CPU budget before ModelGO. Prefer A PP2cap16 prefillmicro2 when its admission/memory gate passes; save exact failure snapshot before cleanup. If capacity gate fails, stop A and use already-qualified B PP4cap32, not blind floor lowering. Do not change T128/256 guards.

A 33cards C1/2/4/8/16 official32 matched AR/state, fixed/adaptive atC16; B 65cards C16/C32 AR/fixed/adaptive. One bounded load per selected topology, OFF cache throughout, same tokens/caps/task IDs and serving clock. Record correctness/EOS/caps, TTFT/ITL/userfirstlast, correctEOS commonwall goodput and acceptance/verify costs; bounded queue can stay resident briefly for analysis, then stop within3h/2h respective budget. A and B are alternative capacity/performance paths, not permission to unconditionally run both.

No extra draftcache/head/channel grids; no tokenidentity qualitygate. Promote only with observed task/finish behavior and useful user gain;32tasks remain pilot. New loader remains separate CPU-only.

**K105B · kimi_continue_92 · COMPLETE_SCOPED_REAL_MOE_EVENT_DIAGNOSTIC**

Use existing conditional16card parent373 n17/n18 one-layer run with exact c390 host/bridge linkage and defaultsOFF; keep no extra synchronization. Collect route histogram via pinned async copy and original Work completion, event spans and wall; use source/stream identity to separate arrival waits and compute. Generic FakeDriver386 alone did not qualify runtime. Retain historical16row scope; do not call it currentC32 criticalpath.

Use bounded result to choose one target-side intervention and validate promptly in next justified same-source Model. Current target verifier occupies about90% cyclewall but nested timings overlap; do not sum costs, infer pureGPUbusy, hardwarepeak or claim MoE dominance before measurement. No repeat diagnostic merely to enlarge a grid.

**D105A · deep_continue_92 · ACCEPT_MODE1_SINK_PREFILL_OPT_IN_STOP_REPEAT**

Retain immutable9330b4f sinkbroadcast candidate as opt-in for future justified integrated runs, with current mode1 QK/PV/layout/provider unchanged.1079 CSA component -15.74% wall;1083 actualModelPF +14.89% throughput and B32 inclusive output +4.89%, while commondecode -0.13% is no gain. Stop repeating this same component or coldloading Model just to reproduce it. Further Deep optimization must target a measured decode dependency from existing profile or one necessary instrumented next run, then one fullModel validation.

B32 matched32/32 correct andnaturalEOS botharms; not formalno-regression. ExtraB16 next16 ON-only15/16 cannot join paired denominator. Maxuser~2.09, >15goodput0; no SLO claim.

**G105A · kimi_state_only_replay_93 · FIX_COMPONENT_CLEAR_THEN_QUALIFY_ONCE_AND_ADVANCE_MODEL**

Keep M16. Apply only missing modern_glm_fp8_runtime_clear_r809(owner) to componentclose before runtime teardown, preserve realModel existingclear and unchangedmath. Require counters created==destroyed/live0, finite/read-only/canary, fallback and <=32 lease bound with real owner/stream/context. CPU then alreadyconditional singleGPU n14; original704/709 remainFAILED6 and0qualified. Use symbolized Store::~Store nonemptyowner evidence, not speculative lease/TLS cause.

After normalexit/audit0/postempty and actual projection-chain exposed gain, proceed promptly to one same-source matchedModel comparison in justified load; do not keep repeating4M microbenchmarks or coldload to retest M32. Any failure leaves performance unqualified.

新Model源计划仍须独立冻结 exact source/resource 与CPU预算：PP2 cap16为33卡、PP4 cap32为65卡；优先可行的一条，保留有界resident queue便于一次加载内分析，3h/2h上限前停止。每模型独立owner，避免无必要串行。

T64限制贯穿owner transaction、provider、workspace和tcounts；仅升frontguard会越区。T128/256需完整改造，禁止盲升。4GiB是规划floor，421 afterfinish通过而C8PF失败，真实失败低水位未知。last-row head已在104证伪为新增机会。新loader仅CPU准备，未实现ownerreadback/mode/abort CAPI前不能进Model。

原GLM704/709 FAILED6、post705/710 empty仍为0资格轮。父任务提供的符号定位是Store析构时owners非空；修复组件clear，不能凭空归因lease或TLS。历史16row MoE与CPU证据均不能冒充当前C32性能。

来源、SHA、完整指标及父任务状态边界见相邻 input/current JSON。本review未改Git、ledger，未运行GPU、联网或访问protected910384。
