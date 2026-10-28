# r171 最短设备分配/生命周期卡（待单独审核，未提交）

建议只做一次单卡、单进程、新Native的受限验证。无需再跑regular/exact差值，也不重复constructor767。先复用r168共享helper的提取方式，将r170原Experts packed分配/5376leaf绑定代码提取为同TU共享helper；真实Experts构造保留原comm guard并调用该helper。新probe CAPI放在**同一个MoE TU**，调用该helper和r170实际`shared_workspace/Lease`，复用实际r170 cache.o。这样不在另一DSO重新实例化一套lease registry。预计仅新增一次HOST MoE+probe编译/链接；锁实际r170 CPU receipt/Native/对象和新source SHA，以dladdr核probe及cache factory真实来源。r170生产包不覆盖。当前只准备本卡，无SSH/Git/CPU或GPU执行。

## 一张卡能直接覆盖什么

能直接同时持有92组真实exact packed bank（368个storage、494,592个Weight leaf别名）、完整93层缓存（24MLA+69KDA）、同owner的92个workspace字段。packed逻辑45,201,752,064B，cache逻辑1,086,011,136B，workspace逻辑3,878,656B；合计46,291,641,856B，约43.112GiB，**只是这些持久分配的逻辑子集**。每个bank都从生产共享helper真正创建，不能用一次分配乘92充当设备实测。FP4 bank保持未初始化且从不读取内容：不加载权重payload、不执行forward。

cache factory允许null communicator：`validate_cache_runtime`验证owner/device及TP/PP几何，不查询/使用通信句柄。probe仅给它原TP32PP1、rank0的分片几何，用于产生H3 cache；明确标`allocation_geometry_only=true`、`communicator_created=false`，不伪装真实TP32执行。若实际编译的factory新增comm依赖，就停在CPU并报告，不能用sentinel或单rank comm替代。

单卡只在实际free足够时继续，每次新增bank前保持4GiB floor，并额外保留2GiB给完整cache及诊断。读实际total/free，不假设物理SKU。固定目标92组，若中途容量不满足，记录实际完成数量后安全退出，不静默降成小样本并宣称通过。92组bind map只占主机内存，给16GiB host上限即可。工作空间验证输出与readback缓冲控制在各8MiB内。

## 一次进程中的固定顺序

1. 创建owner A，记录Runtime前/后实际mem_get_info；调用生产shared helper92次，每次保留真实binding和四bank，核完整shape/stride、4storage/5376leaf及末offset。记录第1组和第92组端点，最终92组全部并存，避免外推。
2. 调r170实际Kimi cache factory创建完整93层vector，核原24/69分布及形状；调用owner stream fence。对24个exact MLA bank逐字节分块D2H（8MiB host buffer），检查全零；同样检查KDA零初始化，保留原factory的zero-fill与发布边界。这里读取的是新cache，不读未初始化FP4 bank。
3. 构造92个仅保存Tensor字段的持有者，每个调用实际r170 `shared_workspace(A)`，验证同一TensorImpl、data、allocation_owner和registry generation；**这不是构造92个MoE对象**，而是验证原MoE构造所用的相同workspace赋值入口。92字段并存时只应增加一个workspace。创建owner B并取得其workspace，两owner同时存活，核不同storage/control-block/generation；两个Runtime均真实创建，无comm。
4. owner A正常lease：在实际Runtime stream排队scratch全长memset、scratch→独立输出的D2D读取及cudaEventRecord；不在lease退出前同步。退出后立即eventQuery必须success，再D2H核输出pattern。异常lease同样排队后抛自定义host异常，使真实Lease析构在unwind中drain；捕获后eventQuery与pattern核验。另在活动lease内尝试嵌套acquire，确认非重入拒绝且外层仍正常完成。
5. 用B的workspace向A申请lease，应在任何新GPU提交前拒绝；A/B分别正常工作，验证隔离。相同Runtime的stream不可通过公开API更换，跨stream防护保留源码证明，不侵入私有字段来伪造测试。失败drain的terminate路径只保留源码证明，不注入设备故障。
6. 释放B及其workspace；释放A的全部bank、cache和92字段，记录对象销毁端点（不trim）。再释放A，核弱owner/weak TensorImpl均expired，记录Runtime销毁端点。创建owner C，实际调用shared_workspace，确认新control-block/generation及旧registry记录可清扫；不要求raw地址不同。最后关闭C及event，正常marker、进程COMPLETED0、post KFD空。

每阶段free必须>4GiB；保留实际used端点与最低free采样，标为**采样/持久子集端点**，不称精确连续峰值。Native pool reserved仍UNKNOWN；Runtime销毁后相对进程初始used的残差单列、归属UNKNOWN，不称leak或完全回收。可复用r168a现有单arm控制框架，parent374/n01、一张可见GPU、inner180s、Slurm4min、CPUkeeper约8min；不要求IB/CCL资格。任何失败只清理本次真实句柄。

## 通过后仍然没有证明什么

它证明实际r170持久bank/cache/workspace子集能否同时分配、cache是否清零、共享lease是否在正常及host异常路径完成同stream消费者后才复用、跨owner隔离与弱所有权释放。没有构造全93层参数对象、router/replicated上下投影/embedding/head、完整AttnRes、模型安装/转置或任何真实forward。

因此正常算子T512/T64临时空间、SDK/GEMM/attention/router高水位、load/finish staging和双份存活、CCL首次及稳态空间、全模型参数物理粒度/alias账仍UNKNOWN。单卡不含真实32owner CCL，不能据此fullModel GO；43.112GiB也不是58.906GiB全部模型。即使全部通过，目标仍UNKNOWN_STOP，只允许更新持久子集的实证容量账。

**每次I96新增sync的速度风险继续保留。** 本卡不计吞吐、不给加速结论；正常/异常事件测试不是I96算子或Model性能验证，不为去fence再加实现扫描。
