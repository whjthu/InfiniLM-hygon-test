# GLM TP16 完整模型入口

这是完整78层checkpoint、16个真实owner、TP16/PP1、两道固定原题的首次入口验证；不是另一次SUM/Layer0测试，也不是完整质量或在线SLO验收。复用CPU1087的Native22345a与876两leader framing，不重新构建。TP8默认和旧benchmark不改。

资源：parent910371的a02r3n15,16，每节点64CPU/224000MiB/8卡；keeper parent910407/a14r2n20为2CPU/8GiB。C1/完整32768/显存8GiB floor。GPU14400s、Native工作13800s、inside硬停14100s、keeper15000s均以原epoch计，加载10800s、构造协议1800s。独立fresh/post及清理也在keeper原预算内。失败不重启，不减context/floor/输出预算。

先ordinarypub，再HOST同容器preflight验证完整ENV、依赖和私有原输入；新namespace/intent/标签无记录后唯一launch。GPU前逐节点PID/KFD/PCI/端口/实际membership/活跃step去重。launch --preflight-only不调用Native/Runtime/SDK、不打开权重；Torch import会加载相关动态库，ldd子进程初始化仍UNKNOWN。通过后相同publication/manifest的一次真实提交，原HOST/PID/startticks保存。

两道输入是原固定256题中的ordinal0和128，分别完整8shot GSM/5dev MMLU，复用原CPU918 token IDs。每题原4096生成预算、自然EOS或cap，错题继续保留分母，不加题/补题/重编码。完整load完成后两题全层forward、root greedy/world control广播、两次before/after reset、正常destroy/weakRuntimeexpired。typed handle绝不传legacy CABI。

create/load/两题/quiescent/closed共6个原协议barrier。各rank原key集合覆盖78层，CPU张量强引用存活到同步load_batch返回。所有16个生成ID hash与root一致。输出全留私有R；Git只保汇总和SHA。后台5ms driver采样与CAPI边界8GiB检查不构成未采到峰的硬上界。没有emptycache/峰值重置；不将普通全局Context传旧trim。

keeper保留全部accounting可见性观察，最多4次终态row读取、2秒间隔，无二次launch。错误集体状态不明时不重新进入Native collective/destroy，由原owned Slurm failstop清理；post独立验证原PID/startticks、保存UIDsocket inode、KFD和已知端口。旧876 keeperFAIL保留。GPU正常不等keeper已终态成功，二者必须原件分别消费。

本地check_checkpoint_CPU.py只测新loader的78层覆盖、64位指针/源生命周期，非实际Native数值。真实入口通过后才接参数队列、宽质量和同口径性能；backend采样时钟不冒HTTP/clientSLO。

本v2只修HOST预检的证据复用和进程归属缺口，不重编/改模型数学。v1预检90秒失败及原同HOST processes=[]保留。21e17发布的一次owned-PGID诊断实际57.75秒通过同ENV/容器metadata、Torch CUDA未初始化、tokenizer和ldd，原45.8秒Torch导入单列；不能反推旧失败具体阶段/NFS。新v2preflight实际核自己的source/CPU/输入/脚本，并校验该诊断完整原stdio/阶段顺序/缺符号及ENV/Native SHA，复用既有成功环境而不再重复昂贵import。此资格仍不等Model/GPU/fit通过。
