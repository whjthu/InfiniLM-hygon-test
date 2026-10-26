# r120：独立 TP32/I96 layer1 MoE 源码准备

只新增一个 HOST adapter，链接现有 Native305726；SDK6d5a 的通用 I96 TU/对象映射沿用 r118。没有编译、GPU、权重读取、SSH、Slurm 或 Git 操作。不是 full93、输出 head、KDA 或投机回滚实现。

`TP32_I96_MoE_r120.cpp` 沿用 r9125 的 CommunicationGroup、原生 module/Weight 加载与 owner drain。每节点8个实际 Runtime owner，TPnodes4、PP1，总 rank0..31。构造后检查全部5376个专家物理 tensor 的 I96/packed48/scale3 形状与5384个 required keys。所有全局源 descriptor 仍交给原生 Weight::select_source，禁止先分片再二次切片。

prefill setter 和 tile setter 都固定0；private workspace、I192/serving/pair 开关固定0，数学模式0。T8/16/64 都走现有 generic impl1；每次 forward 检查 I192、旧 serving、W2 pair 计数无增量。该证据是源码分支加排除计数，尚未新增通用 provider 正命中计数。没有调用 I192 kernel 处理 I96 指针。

输入沿用历史6110 teacher0 layer1 的16行 BF16。T8取前8、T16保留原16、T64重复4次；consumer 实际统计二进制 distinct rows，不声称16个独立上下文。旧证据指出可能只有8个不同 latent rows，不能冒充当前真实用户激活。本轮不添加构造 routing fixture。

每个 T 两次 warm、四次普通重复。32-owner F32 sum32 barrier 在计时外，计时为完整单层 forward 加 owner stream drain；输出持久化、有限性与 input/canary 检查在计时外。重复差异及跨rank摘要仅诊断，不采用 token/中间值逐位质量门。不设 A/B 倍率或整模型 TPS 声称。

权重 consumer 复用已通过CPU657的只读 mmap helper，保活 safetensors/Tensor 到所有 native load/drain 返回后，避免旧654的16GB clone+重复SHA方式。现在仅复制其源码/头校验表，不读取权重。CPU build entry 从现有 r9280 的 runner 命令提取相同 flags，仅替换 adapter 源、输出和依赖文件；不重编 native、SDK、GPU TU。实际 compile/ldd/依赖闭合仍须 root 发布后运行。

源预算：expert packed+scale491323392 B/rank。旧TP16实际单层参数1114778112 B中复制115616256 B，按TP32分片推导615197184 B/rank，属于保留源码算术而非 allocator 峰；derived融合权重可能额外存在。当前 ctor 最小 scratch12288 B；最大T64 generic activation196608 B。单层 named input/output/routes+scratch约1.12MB，不能代表完整 temporary peak。规划参数上界2GiB、临时上界1GiB是宽裕待验证的组件预算；CCL/driver/allocator charge未知，保留原4GiB实际floor，不能据此宣称物理peak已测。它与全93的57.894GiB参数估计不同。

将 source manifest 与 producer index 发布后，root 可先执行 CPU entry。CPU阶段不读模型payload、不dlopen、不创建Runtime、不用GPU。之后应单独冻结运行manifest，填新鲜IB地址、CPU wrapper SHA与同一四节点/父allocation的资源证据。此包不含 srun、自动重提或作业终止。future keeper 需32卡独占前审计、内层720s/外层足够清理余量、同句柄失败全体收尾、32 normal closure/Slurm terminal/post-KFD；任一失败不报成功。不能触碰379模型、371 draft或910384。

必须另审的实际环境事项：当前r9280 receipt runner flags/头ABI、helper依赖可解析、CPU wrapper真实dladdr、fixture实际distinct度、4节点32 rank transport与输入源一致性。未通过CPU编译前，本包是 reviewable source，尚非 GPU-ready。
