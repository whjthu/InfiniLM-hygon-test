# K127A 容量组合原型 r170（仅CPU准备）

基于实际CPU522 / r167 exactpacked Native `eddc95e620da4a3de06f03802ab8ba7c1bcc9cc4a27fd963b9c175dc3262b6d6`。仅改两个HOST TU：Kimi专用cache allocator、MoE workspace所有权/lease。复用其余32个已锁object及I96 HIP/provider，不改Core、通用allocator、类布局、GPU数学或共享DSO。没有GPU/Model提交器。

部署 `/refactor/incoming/Kimi-capacity-exact-cache-lease-r170`；root审核发布后，在既有masked CPU keeper运行 `bash CPU_inside_r170.sh`，660秒硬界。新输出目录 `/refactor/candidates/Kimi-capacity-exact-cache-lease-r170` 必须不存在。编译两个HOST TU、严格链接/nm/ldd、真实MD及全部父object SHA核验；不创建Runtime、不导入Torch、不dlopen候选。

## cache exact 已证明的范围

只对原Kimi cache分配器的TP32PP1、93层、96heads、head_dim192、pages68、block256、BF16 MLA路径使用私有exact factory。它复制原Paged factory的rank验证、shape/stride、zero-fill、Trace、失败drain和publication边界，唯一allocation调用改为`Tensor::empty_parameter`；CPU逐字符逆变换核对原factory。原KDA cache分配和三cache vectors返回路径保持原字节。其他几何走原factory。reset_cache仍沿原owner/topology验证与vector move，没有另加一次先普通分配再复制的双份bank。

这消除24个MLA bank的source pool-class差648,019,968B；底层物理malloc与完整峰值仍UNKNOWN。精确分配不等于已完成GPU清零/读写资格。

## scratch lease 的源码安全论证

- 原每层Tensor字段保存同一个workspace Tensor；registry仅保存weak Runtime、weak TensorImpl及元数据。map使用Runtime共享控制块的owner_less身份，另分配单调generation，不用raw Runtime地址作键。实际Runtime.hpp `c127b4cf...`公开继承enable_shared_from_this，TensorImpl公开allocation_owner/shared_from_this；不新增Core接口。
- 所有构造/消费先assertOwnerThread，graph必须inactive；lease核同一allocation owner、TensorImpl、stream、generation和busy状态。跨owner、跨stream、同owner重入直接拒绝；不等待一个可能递归持有的锁。并行的不同Runtime拥有独立workspace。
- actual provider163的两个memset和packRoutes/buildJobs/W13/W2四个kernel都显式使用传入Runtime stream（原HIP源SHA已锁）。lease在I96原geometry检查后、首次provider写之前取得，并保持Runtime+Tensor强引用直到同stream成功sync。析构先drain，再在registry mutex下解除busy；drain失败terminate，不发布可复用workspace。因此层间、下一请求和replay无需假定host调用顺序就能覆写旧scratch。没有消费者在其他stream上继续使用它。
- generic/prefill路径不收到这个I96 workspace，I192/旧tile consumer均有不匹配I96的原geometry guard；其行为不改。MaxT64 workspace大小及offset不变。registry不强保留Runtime，不形成TLS/Runtime生命周期环；最后module Tensor释放后保留的weak记录在后续构造时清扫。
- **每次I96调用新增显式stream fence，改变调度，性能UNKNOWN。** 这是安全容量原型，不能标Model或speed GO。GPU生命周期实证尚未运行；未来是否用event/有证明的FIFO lease减fence是另一项评审，本包不扩散实现。

## 容量与停止条件

58.906441GiB旧logical baseline不能减11.629GiB packed class差。cache exact与单owner一个4MiB scratch相对92个的source设计额外节省分别为648,019,968B和381,681,664B；条件性opaque前余量1,154,358,724B（1100.88MiB）。参数/KDA cache实际分配粒度、driver/CCL重叠、load/finish与T512/T64算子高水位仍未闭合，fullModel fit维持UNKNOWN_STOP。CPU编译通过只证明新私有Native可链接，不证明设备实测节省、完整生命周期或端到端质量。
