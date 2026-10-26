# Astra 104 冻结评审

本次仅消费三组新合格完整实验：Deep1073、CCL3553、CCL3561/3563/3565/3567 四臂整体。三份 canonical 及支撑文件已自行 SHA256 固定在 input；channel 四臂不拆成四轮。无 Git、ledger、GPU、网络变更。

- **保留 RCCL auto，停止盲扫。** 实际日志为32/1/4/32 channels；顺序 auto_a/c1/c4/auto_b，并非ABBA。BF16 229376B 为0.18104/0.49448/0.23032/0.18104ms；64MiB为2.83608/33.34982/8.80664/2.83728ms。每点仅10个postwarm，16owner均normal、逐臂postKFDempty。只支持否决此拓扑下cap1/4，不能推导Model收益、NIC物理峰值或GDR。
- **Deep推进一个真实形状组件。** 1073完成8卡13m05、audit0、post1074empty。latent占inner约82.17%；CSA21层约3.004s、HCA20层2.178s、SWA2层0.209s，均为逐device顺序层之和的均值，不跨卡相加。已有mode1批量QK/PV。下一步验证现有r9040 per-run sink广播替代per-query复制，真实B4/T1681四对；50397复制→6467runs不是性能收益。组件有效即进入下一次合理的完整Model验证，失败则停止。1073同版本32/32正确EOS、cap0，但用户均速1.10055 tok/s，>15达标0/32。
- **Kimi last-row head提案被源码反证。** causal_lm.hpp 的!sample_all_positions、offsets和packed条件下，已先取每请求end_offsets-1的hidden，再计算head；model.cpp文本路径直接调用它。全hidden仍返回，specverify allpositions仍全行。不要再实现同一优化；仅当已有row计数/链接源证明未命中时修复flags/shape，且另开私有冻结版。4GiB是规划guard，421失败时最小值未落盘，不能归因为head临时峰值或OOM。
- **Kimi当前PP4实验继续冻结运行。** 按parent报告64target+1draft、all64 cap32容量32.03GiB通过，实际C16/C32 AR/statecache OFF/ON结果尚待。先完成32题、答案/EOS/caps/用户时钟和接受率；下一瓶颈优先看target verifier-MoE与流水线依赖等待。复用现有telemetry，禁止热改和重复tokenidentity组件网格；阶段耗时包含等待且互相重叠，不能相加作为compute。
- **GLM保M16，先真实单卡Projection链。** CPU3781/27复用仅CPU资格；实际FP8Projection上下文、有限值、<=32租用/fallback、复用和销毁生命周期通过后再考虑下次完整Model。禁止假context和仅为重复MB参数冷载16卡。
- **loader只做独立准备。** r9212d缓存优化针对_infinilm.so原方法，每flush反复构造整stage约24万key；须保证版本/alias/失效和异常生命周期。CPUmanifest成功不能当loadgain，不混入当前PP4。下一次合理加载测端到端时间、host/device峰值，再验证Model。

421历史口径保留：C1一题AR4.96/state17.07正确EOS，随后C8 PFguard失败；afterfinish5.109GiB通过、lastcasestart4.568GiB，失败snapshot最低值UNKNOWN；不称afterfinish失败或0warm，整矩阵计0轮。当前目标未完成，物理BF16硬件峰值未知。
