# Astra 126：停止微小计算改动，优先验证TP32容量

决定：FP4 direct1181停止；W2 quad4/1184停止，默认pair2，不做L47、Model或相邻参数扫描。TP32/I96保留T16/T64/Gaussian组件收益，T8原漂移条件未过的事实不改写。当前最短的新实证是r167四个持久packed bank的精确分配；同时完成已在运行的r162a加载与C1/C8 pilot，不能把RUNNING当成整模提升。

本次只读本地repo125审阅、implementation、ledger round2、1181/452结果、9503原记录、1184 compact、r167源码及bound DSO证据；仅在v写本审阅，未SSH/Git/Slurm/启动GPU，未改repo/status/ledger。

## 125落实与时间

- K125A已执行并及时止损：1181六组L1配对full-MoE吞吐−14.82%～−17.73%，W13−27.49%～−29.69%，OFF漂移<1.2%、NRMSE0；不继续L47是正确执行。CPU512/ISA不是收益，实际负结果才是这一轮结论。
- I96/452完成实际32rank、IBext、正常退出与postempty。T16为1.270/1.272×，T64为1.409/1.413×，Gaussian64为1.316/1.320×。T8为1.156/1.160×，但首block OFF漂移3.2325%超过原3%，不能宣布全部shape通过，不能为刷过门单独重测。
- W2/1184完成16owner、IBext、正常退出/postempty。T8稳定第二block full-MoE−1.347%、W2−4.611%；首blockfull+3.818%伴随5.426%漂移不算收益。T64历史full+0.220/+1.943%，Gaussian+0.564/+1.734%；所有六block未过原full-MoE与W2各≥5%条件。NRMSE0不构成推广理由。
- r162a的515旧GSM32字段问题已修，CPU454四节点预检过；实际target456/draft598/keeper519在9503时仍RUNNING。四节点加载1186～1310s、18.27万～23.67万keys，未完成pilot。继续同一实际句柄，不重提、不热改。加载20分钟本身尚不能判断cache改动有效或无效。
- 125后这三轮是1181、452、1184；CPU520编译28s及515失败不算性能轮。现在已在做真实实验，主要长时间仍是r162a的Model加载；不要再把资源准备、编译、轮询写成收益。Deep/GLM的125离线成本定位未见本轮新产物，不能登记完成。

## 下一步卡（每模型最多两张）

**K126A：收完当前r162a，不开启第二个Model。** owner为Kimi原owner/root；沿用原32target+1draft和456/598/519。保持cacheON、W13OFF、pair2、原20请求/16独立题pilot。首个有效结果是加载结束后的per-owner registry/flush时间与首个C1自然EOS，剩余时间UNKNOWN，历史总加载25～30分钟只能作为参照。沿原边界继续到6case/退出，完成即读终态并消费结果；不等待下一审核或Git窗口。看当前剩余时间来自registry、CPU页错/IO、其他Native阶段中的哪一项；owner CPU时间与嵌套inclusive桶不相加为wall。固定ON不能因果证明旧1180～1408s由反复state_dict造成，不能声称冷加载AB加速。pilot仍检查合理性、正确/EOS/cap与用户时间线，后期benchmark无明显掉点，不要求exacttokens。

**K126B：立即r167 CPU闭包，然后最小真实分配对照；owner为Kimi原owner/root。** 假设四个packed持久storage采用regular pool时有实质size-class填充，empty_parameter能去掉这部分。复用CPU517其余33objects，仅1个HOST TU、原I96 kernel/math不动；已知路径/依赖/adapter绑定修复由owner在原有界授权内连续做完，不逐条等root。

最短GPU证据使用空闲节点的一张卡/单owner，在各自fresh进程中经实际Common Runtime、实际I96四bank构造路径比较原Native与新Native：同一device、同四shape、同生命周期，记录构造前、四bank后、原expert narrow/squeeze views后、释放/Runtime退出的requested/reserved及mem_get_info。验证四个unique storage、5376个leaf views的别名范围和释放成立；不需要全checkpoint、CCL、全Model或旧generic constructor矩阵。若原入口必须创建分布式上下文，不伪造TP世界：给实际bank构造增加最薄的只测分配入口；准备预计10～20min，GPU实际测量数分钟。旧r163 adapter绑旧Native，必须重绑并核真实符号归属，否则测量无效。

每层设计差135,725,056B=129.44MiB；92层12,486,705,152B=11.629GiB，是待实测的分配策略账差。用fresh进程避免旧pool缓存遮蔽差额，不能只比较Tensor逻辑nbytes。若实测无对应reserved差或别名/释放异常，停止容量晋级，定位这一处差异；不盲目重复constructor。测到差额后，将原实际完整参数/cache/scratch对象清单更新为统一bytes的峰值账，并仅为尚缺的真实高水位做一次有界验证：包含cache classpadding约0.65GB、I96 scratch新增约0.33GB、装载瞬时重叠及driver/CCL/算子工作区，保留原4GiB floor。单层节省不能证明整模fit。

特别防止重复减法：旧58.9GiB若是逻辑requested账，11.629GiB是此前遗漏的额外regular padding；exact是消掉额外项、回到逻辑账附近，**不是58.9−11.629≈47.3GiB**。剩余opaque peak未知时仍UNKNOWN_STOP。只有容量闭合后才提出TP32整模C1/C8因果比较；不趁当前456驻留改变拓扑，也不把同TP32 generic对优化I96的组件倍率当作TP16PP2对TP32整链倍率。

I96 T8不要求先单独重跑才测容量：容量改动本身不依赖T8加速。保留原T8未资格状态；将来必要的真实模型试验可保持T8 generic分支，单独评价T16/T64优化，不能据组件T8把C1收益写成已知。新storage若需要一次单层数值/生命周期验证，可复用452输入和已测kernel，只回答此次分配改动是否影响行为，不重跑整套性能矩阵。

**D126A：用闲置8卡补一次有边界的C1因果诊断，代替再次写离线待办。** owner为Deep原owner；保持ac937、n05/TP8、默认已有数值路径。125的“离线选择新机制”尚无产物，而Deep真实C1仍缺；允许把这次加载视为明确诊断任务：一次约7～8min加载后，4个现有真实QA自然EOS、同实际C1形状短warm/稳态窗口，复用现有owner/event接口，只分清Native算子/collective/host间隙的主要时间。预计已有接口可复用时15～25min得到结果，准备/排队UNKNOWN。不得再扫B4/8/16/32、早decode或F100旧synthetic；不能只重复总TPS。若现有探针无法区分主桶，先修最小探针再加载，禁止花8分钟加载后才发现无可行动数据。结果只决定一个下一kernel/通信候选，不宣称本轮优化。此项对125的暂缓作一次明确调整，理由是两轮后仍无C1归因且有独立空闲资源，不是降低质量要求。

**G126A：保持full+deferON，针对C1准备一次最短可行动诊断；无新候选时不做chunk/batch扫描。** owner为GLM原owner。784已经给出C1用户5.01tok/s、约0.199s ITL；目标是这个稳态Native路径，不能再优化first-token调度代替它。先核767的原捕获shape/桶是否覆盖C1；覆盖且无新增机制则复用结论、不重载。若767未覆盖C1或现有大桶不可行动，复用其source把观测限于一个实际C1窗口的主要算子/collective，保留邻接OFF检查后只跑该短诊断和基本正确/EOS。准备建议20min内形成具体入口；GPU首判定为一次实际加载+约3～5min窗口，加载/排队当前UNKNOWN。若需要重建宽profile体系，暂缓该线，把资源留给K126B和D126A；不为消耗空闲卡启动无假设作业。

## 收益与硬件边界

W2逻辑重读/CTA减少没有转成实际wall收益，不能反推“已证明HBM瓶颈/非HBM瓶颈”或猜cache命中率、occupancy。FP4直接解码的静态指令改写同样不能预测动态周期。保留已证实的负结论即可，不花更多作业解释不影响下一决策的微观原因。

仍用同一计时范围S=1/(1−f+f/s)。即使把W2某区间2%加速带入条件整链份额f=0.21，也只有约1.004×；该f只是示例，不是当前整链实测，足以说明不能用这项支持昂贵Model重载。I96 1.41×仅是同TP32组件对比；TP32的全层关键链份额、32rank通信和PP改变后的收益未知，不使用333TF BF16 dense峰值预测scalar FP32/FP4性能。r167是容量改进候选，显存节省也不自动变成tok/s提升。

已知边界不变：不触910384/n17/n18、不读nonce/rawargs、未知GPU状态不重播、不引入GPU F64。审批/路径/依赖已明确的常规修复顺序完成；未知执行、语义或数值问题停定位。126结论落实到原卡和实际句柄即可，不新增框架、报告层或门禁。

## 126补充：r168与已存在的观测入口

r168所述共享helper方案符合K126B的分配证据范围：实际ctor保留tp>1必须有comm的检查，ctor和allocation-only CAPI共用同一四bank+5376views helper；regular/exact两私有Native分别fresh进程，验证source等价与dladdr。这样测的是本地I96 storage分配、别名与释放，不是TP32执行、collective或整模fit；无须伪装单rank comm，也不必为此重跑MoE forward。这是对方案的认可，尚未把未读到的r168实现标为已验证。

以下路径均以 `C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/` 为根：

- Deep当前ac937加载/forward基座：`code/bw1000/resume-20261026/Deep-C4-independent64-private-r9278/modern_Deep_independent64_r9278.py`；`Deep-real-batch-frontier-r158/bootstrap_r158.py`明确复用这一基座，`manifest-r158.json`锁ac937。保留它，不用旧9234的Native版本替换当前库。
- Deep最短现成观测函数：`code/bw1000/resume-20261026/Deep-C4-independent64-private-r9278/deep_owner_consumer_r9254.py` 的 `bind_owner` / `owner_forward_once`（第5/24行），调用ac937已有 `deep_owner_arm_r8900` / `deep_owner_read_r8900`，覆盖8×43行的16个nested阶段；owner版本不硬锁B4，先用于一个真实C1 decode窗口。不要把该文件的 `inner_owner_forward_once`直接当C1版本：其validator第42行硬锁1681token/4requests prefill。
- 更细decode consumer可复用 `code/bw1000/resume-20261026/Deep-decode-profile-private-r9234/deep_owner_consumer_r9234.py`，它已有读取失败前保留scalar packets、关闭window和几何绑定，但第40/54行显式只允许T32或T2..7；改为明确支持T1几何后才调用。对应 `Deep_owner_prefill_r9234.py` 第47–49行selector目前只抓B32及2..7尾部。Native inner header位于 `code/bw1000/cache-pp4-and-deep-profile/Deep-attention-inner-private-r9193b/deep_attention_inner_r9193.hpp`，Frame只拒零tokens/count，源码未排除T1；当前ac937具体T1分支覆盖仍须按真实调用确认，不能用Python放宽断言冒充Native已覆盖。
- GLM767 **没有C1 DECODE profile**：`code/bw1000/resume-20261026/glm-critical-observer-r131/glm_sparse_observer_r131.hpp` 第37–39行仅选择decode T16/T15和PREFIX B8/B1。最终全16owner证据是 `code/bw1000/resume-20261026/glm-critical-retained-export-r137c/G119A-all16-final-r137c.md`；其中B1 PREFIX不能当C1 decode。
- GLM恢复入口用当前784的 `code/bw1000/resume-20261026/GLM-smallconcurrency-chunk128-r160/glm_arrival_probe_r160.py`（已有C1-first8、真实EOS与pipeline时间）+上述r131 observer及同目录 `CPU_build_r131.py`。只增加有界T1 decode selector及匹配coverage解析，复用原owner/stream/收尾，不再跑T16/T15/PREFIX旧selector矩阵；当前r160明确全程关sparse observer，不能只改Python打开就获得T1样本。

