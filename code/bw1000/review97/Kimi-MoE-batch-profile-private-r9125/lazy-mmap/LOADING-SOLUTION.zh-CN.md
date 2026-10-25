# MoE 单层加载诊断与替代方案 r9125

654 因加载不对称在原540秒边界退出。n14 CPU权重准备41.724秒后等待对端，n15未完成；post656两节点KFD均空，无有效MoE性能结果，也没有原生多batch收益结论。

当前加载器每节点取同5384个tensor，总16,102,205,952字节；对每个tensor clone/contiguous，再numpy.tobytes+SHA，全扫复制后才让原生Weight::select_source按TP rank选择。去掉这些准备期完整复制和重复全量hash有明确源码依据。41秒与540秒差异仍包含未控制的冷热缓存/节点状态，不能归因于clone一项。

最小兼容修改是 mapped_layer_weights_r9125.py：safe_open上下文和源Tensor始终保活，保留原完整shape和原连续strides，不clone、不强制额外contiguous、不全量tobytes/hash。原生select_source先按rank narrow；W2因此出现非连续CPU view，与现有加载器实际行为相同，继续走原copy_from。禁止将已分片shape直接喂旧select_source导致二次切片。上下文只能在native复制和所有owner drain完成后释放。

CPU657在n14/n15注册同5384个真实tensor分别1.201/1.708秒，RSS约0.94GB，GPU未初始化。只抽查少量首字节，不是16GB全读驻留，也不是完整权重重新SHA验证。

TP16每节点8rank真正逻辑并集8,108,911,104字节，其中复制参数115,616,256字节可在节点共享；每rank实际1,114,778,112字节。完整按key范围见actual-TP16-ranges-r9125.json。W2按列切片会触达大部分4KB页，不能承诺物理IO也恰好减半。

run_prewarm_CPU_r9125.py在申请GPU前只读预热这些实际页，120秒上限；任一节点未完成就不准入16卡GPU试验，避免另一节点已进入CCL空等。CPU658当前缓存下读取约10.76–10.79GB页范围，n14/n15分别2.426/4.650秒，实际physicalread0.114/0.719GB，RSS约35MB。这是当前缓存测量，不是冷AB；pagecache可被驱逐，不是锁页常驻。应紧接模型加载，并记录完整CPU准备+H2D+post总耗时。

进一步方案是一次生成每rank连续本地cache并在私有HOST加载器中按实际destination shape/dtype校验后直接copy到state_dict()[name].tensor()，保留全部post hooks和owner drain。该方案尚未实现或GPU资格，不改SDK。整模型缓存必须按Slurm内存预算分层/LRU，不能将单层8.109GB无限累积。

历史L1逐tensor hash来自654 node0已完成CPU准备的partial producer，见prior-L1-tensor-hash-bindings-r9125.json；不能称本次fullcheckpoint已验证。657/658的原producer路径和SHA见CPU-loading-terminal-r9125.json。
