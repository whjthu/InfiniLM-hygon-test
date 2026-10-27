32卡 constructor-only 已通过：910375.767 COMPLETED0/46s；keeper910407.465 COMPLETED0/56s；before766/post768均COMPLETED0、KFD空。watch SHA `cb2e0abdb004fef7f2a4fe0682a6b2fc8a68f3a5dc2f2ef8284d5a010092ea5a`。4节点结果原始SHA及完整JSON内容已保存，报告JSON SHA `619be71623016613d04c7f7bed2786db1287c405ba139bf3842cea69996cfc0e`。

实际证明：32 owner正常关闭；69 KDA+24 MLA共93层缓存每卡1,086,011,136 B；H3 KDA/MLA构造形状；5120行本地head与163840行BF16 rank-tag metadata allgather。private Native0f、Common2e14、wrappera891的实际dladdr分组均正确；4节点IBext_v8且无Socket data路径。I96 math0 MoE沿用此前764的独立执行证据，本次767没有构造/运行MoE。没有checkpoint、93层参数构造或forward，也没有H3前向/replay数值验证。

每卡cudaMemGetInfo可见总量实际为 **68,702,699,520 B = 63.984375 GiB**，不能据此单独认定physical SKU。128个阶段样本最高used **2,896,166,912 B**、最低free **65,806,532,608 B**。这不是连续峰值；graphPinned、driver/CCL峰值及whole-run peak均UNKNOWN。释放组件后的采样仍早于Runtime/CommunicationGroup关闭。

|阶段（32 owner范围）|sampled used B|
|---|---:|
|before components|880,803,840–889,192,448|
|oneKDA+oneMLA+head+all93cache|2,805,989,376–2,814,377,984|
|metadata allgather后|2,887,778,304–2,896,166,912|
|components释放、trim后|960,495,616–968,884,224|

已验证的上述组件逻辑字节合计 **1,232,781,068 B = 1.148117 GiB**：KDA29,521,932、MLA43,847,680、head73,400,320、cache1,086,011,136。逻辑字节与sampled used的差额不能直接标为driver或不可复用开销。

全93 WORKSPACE_OFF静态估算仍为 **58.906441 GiB**（参数57.893962、缓存1.011427、92层私有OFF scratch1,130,496 B）。按实际可见总量和不变4GiB floor，所有未计入项合计仅可用 **1,157,422,532 B = 1.077934 GiB**，比名义64GiB计算少16MiB。可用峰值上限为59.984375GiB；这只是容量门槛，不是已经证明的全模型峰值上界。

如果before-components的使用量全部持续且与静态张量不重叠，static+baseline约59.726754–59.734566GiB，最坏owner只剩 **0.249809 GiB (268,230,084 B)** 容纳其余开销。该持续/不重叠条件尚未独立证明。若机械添加constructor allgather后的全部残差，得到60.447778–60.455590GiB，会不满足4GiB floor；但残差包含可复用池/分配粒度/通信等，不能把这个投影当作物理不可能证明。也不能把残差当零。

**全模型fit仍UNKNOWN/STOP**。58.906GiB包含保守复制项，只是有条件静态账本，尚非无条件allocator上下界；完整峰值上界未闭合。WORKSPACE_ON静态60.546GiB仍直接越过59.984375GiB门槛。下一最小工作为CPU源码核算全93真实分配粒度、不可复用bootstrap与池重用关系、load/finalize/forward/capture并存寿命；完整正预算以前不进行全93构造或加载。未知graph/driver/scratch峰值保持UNKNOWN，无性能外推。

本轮仅一次固定watch+4个引用JSON的只读SSH；无GPU/Slurm查询/compile/dlopen/upload/Git/热改，未查询384或n17/n18。原k未stage；kb仅GPU_inside脚本CRLF→LF，实际manifest `c50e160faaf5865555cff7b6e1d30fa615b5718c960833997b0bee5077f1468d`，root出版 `6d59eff7329a3c00459303f87d3bd9bc0fc2cad6`。
