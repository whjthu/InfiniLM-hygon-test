# InfiniLM 上游静态复用核查（2026-10-27，r130）

本次只执行本地 Git 对象读取及已授权普通 SSH Git fetch；在专属 bare 库取四个引用和四个指定 blob，没有执行上游源码、构建脚本、测试，也没有访问集群、Slurm、GPU、上传或修改运行依赖。结论是源码适用性判断，不是 BW1000 性能验证。

## 当前运行版本与上游关系

| 引用 | 实取 SHA | 相对运行 5f91ad 的关系 |
|---|---|---|
| 运行基线 | 5f91ad738583db8f5617aeae53b76523f85d1ec6 | 已含合并前 modern-stack review 结果 |
| main | 1b005e15ad3a0f965935a645d46f3b69abe12314 | 运行基线为祖先，左右独有 0/3 |
| refactor/adopt-modern-infini-stack | f94d83d729cc24d1c87523e9746f62c28566db6f | 与运行版本分叉，merge-base 270feb3e14976ca14c3b012df1d1740f939a941e，左右独有 25/58 |
| feat/hygon-w4a8 | 7ab405849b086be7722eb5ef5458192f79f73e38 | merge-base 8657a943553e2c67ad5f3db3db32931d8c97c3ae，旧接口旁支 |
| dpv4-ep | ef203f929a438c8d1124d3c96af53af63eabe7b7 | merge-base ee4c81071dfd686059bf3e58a5a75c0b24d9bfd0，旧接口旁支 |
| perf/paged-graph-execution | 50ff7ce209ec9191dd5cf2b03a1d9668aec2cd89 | 运行基线为祖先 |

main 的三项后续提交为合并 PR #596、Qwen3.5 norm/TP 修复 274aa5d，以及其合并提交 1b005e1。相对运行版只改变 rank_worker.cpp、attention.cpp、modeling_utils.py 和 checkpoint raw loading 测试四个文件；不能将当前运行版本描述为尚未进入 modern-stack 重构。

main 与运行版的依赖 gitlink 完全相同：Core **2b4f2f20584a69b074f62a455e3de7368e3446c2**；该 Core 的 RT **cd1d48258f0a6b8d9f66111d5d6cb1664205e78d**、Ops **1ca4777610d77e9063fffdb975ab35da8466104d**、CCL **51cc9db31d58aac21643d72f0f77cf5c9ecfaee5**。JSON/spdlog gitlink也相同。证据见 running-dependency-gitlinks.txt 与 main-dependency-gitlinks.txt。

f94d83 删除 third_party/InfiniCore gitlink，内嵌 csrc/infinicore 并构建自身 Python infinicore；scripts/build_infini_stack.py 接受外部 --infinicore-root，再读取该 checkout 的 HEAD 及三个子模块 gitlink并记录manifest，LM自身不固定那个外部Core SHA。因此它不是可直接替代运行依赖的线性升级。原 modern-infini-stack-review 分支当前不再有可见远端引用，但已合入main的历史仍在。

## 已核实的复用候选

### 1. Kimi W4A8：优先借鉴权重布局、shuffle 与 fused 调用，避免重复实现

证据：7ab4058 的 csrc/layers/quantization/w4a8.cpp 与 csrc/models/kimi_k3/kimi_k3_moe.cpp；功能提交 3ea6c23，后续精度修复 ddea55c。

* 通用 Linear 使用 packed I8 权重 [out,in/2]，每输出通道 F32 scale [out,1]；in_features 必须为偶数，fused Linear 的 split_params 仅支持输出行拆分。forward 调 linear_w4a8(input.contiguous(), weight, scale, bias, alpha)。
* Kimi routed_expert_quant_method 为 slimquant_w4a8 时走 W4A8；hidden_size 与 TP拆分后的 local_intermediate_size 都必须能被64整除（源码146–149行）。权重为 packed I8，scale 为每输出行 F32，不能直接把现有 MXFP4 的 block32 U8 scale 当作此格式。
* process_weights_after_loading（196行）对每个expert的w13/w2调用 w4a8_moe_shuffle_，带已shuffle状态；forward（257行）调用 fused_moe_w4a8，激活 Situglu，传入shuffle标志。适合减少Kimi专家计算、权重读取及逐专家启动开销。
* 该分支依赖旧 infinicore::DataType::I8/F32 和旧 Core operators；当前modern运行接口不同。必须移植布局/调用契约并确认目标Ops具有对应Hygon provider，而非整支直接合入或切共享native。未核实GLM/Deep量化格式满足此布局，不宜泛化。

结合root已确认的当前Kimi布局：TP16的local intermediate=192满足%64；TP32的local intermediate=96不满足，故不能把相同实现直接用于两个TP布局。FP4 group32转为I4 per-output scale还会改变数值契约，需要独立量化误差核验；即使形状满足，也未确认BW1000 provider、AITer shuffle/kernel的硬件适用性。

### 2. Deep EP：可复用expert placement；DeepEP通信实现仍为空

证据：ef203f9 的 csrc/models/deepseek_v4/deepseek_v4_moe.cpp 141–172行根据placement.local_num_experts分配权重；开启EP后每个local expert保留完整intermediate，expert_tp_rank=0、expert_tp_size=1。路由还调用deepseek_v4_hash_topk_router，并有linear+hash fallback。

但是该SHA的 csrc/layers/moe/ep/deepep_dispatcher.cpp 四个dispatch/combine阶段仍直接throw“reserved but not implemented yet”（30、34、40、44行）；运行5f91ad和f94d83也一样。**不能把 dpv4-ep 分支名称当成已实现 DeepEP/低延迟all-to-all，也不能据此承诺BW1000通信收益。** 可以借鉴本地expert布局和router合并，降低Deep专家冗余存储；真实通信backend、RCCL适配、跨节点路由与正确性需后续逐项确认。本轮没有核查其他dispatcher实现，也未运行任何EP。

### 3. Paged graph：最接近当前基线，适合GLM PP1 decode准备/传输瓶颈

证据：50ff7ce 的 docs/paged_graph_profiles.md。分支直接从5f91ad演进，提供显式(batch,KV bound) capture profiles、按精确batch和最小覆盖KV bound选图；opt-in graph_metadata_upload用compiler-owned pinned host/device slabs，把多metadata上传合并为一次H2D及顺序D2D，serial lease保护异步源生命周期。

有界profile当前资格为ordinary KV prepare、**PP1**、paged/flash attention、未量化KV、page256且至少两页；prefill/all-position保留eager。不能把它直接用于Deep自定义NSA KV或PP>1的Kimi布局。普通Tensor copy语义不变，upload默认关闭。文档仅声称双A100 Qwen3-14B TP2 BF16/page256/544pages验证，明确不构成BW1000速度与正确性证据。

适合优先移植/实验的目标是当前GLM如果确为PP1、普通paged KV且CPU metadata准备/H2D/graph capture显存是瓶颈，减少重复metadata准备与全池捕获开销。需按实际GLM metadata和Hygon graph支持核对，不应修改正在加载的764进程native。

### 4. adopt-modern 的attention/graph只能做接口参考

f94d83 MHA_VARLEN适配明确注册Hygon；要求FP16/BF16、query head_dim<=256且%8=0、Q heads可整除KV heads，paged block_size%256=0，调用实际provider slot16并验证active implementation。枚举存在不代表BW1000相应provider已可用，也不覆盖Kimi MLA/Deep特殊维度。

f94d83 rank_worker只在PP1建立/使用GeneralCompiler；PagedCompiler只捕获每请求一个query token，精确batch匹配，并拒绝sample_all_positions、target_hidden_states、pixel_values。最终StaticBatchingCompiler为空compile并返回nullptr，static attention host长度走eager。90ddf653性能提交标题中的ChatGLM short graph不能据此声称最终head仍保留该static图路径。

## 对下一轮目标的建议顺序

1. 先沿现有运行5f91ad/Core2b4f2f20隔离优化，main后续norm修复只按受影响模型评估，不需要全依赖换代。
2. GLM符合PP1普通paged条件时，优先研究50ff7ce profiles+metadata upload，减少重复实现异步生命周期管理。
3. Kimi优先提取7ab4058 W4A8布局、shuffle一次性处理与Situglu fused expert契约；先匹配真实checkpoint scale格式及Hygon operator availability。
4. Deep先修metadata并定位实际模型路径；借鉴ef203f9 expert placement/router，不把空DeepEP接口作为通信优化方案。

本目录保存了上述源码的 Git blob 导出和依赖/祖先证据。专属bare通过alternates复用本地已有对象；不作为可搬运的完整源码包。没有执行任何性能测试，未确认这些旁支对BW1000的具体加速幅度。
