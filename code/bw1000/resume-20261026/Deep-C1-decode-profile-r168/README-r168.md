# Deep C1 当前路径诊断
复用实际 ac937 的1178/1168 loader、Native、数值策略，不编译新Native/HIP。只将旧B32/tail profile消费者限定为真实T=B=1；Native Frame本来只要求tokens/count非零，实际捕获仍须每次返回8×43 owner/inner行、正确kind/geometry/事件生命周期及C1计数（FP32 batch attention=0，fallback43）。
一次加载，现有n05固定前四题逐个C1，自然EOS/cap512/context1024。每题decode8采一次owner+inner，7/9是相邻OFF对照；相邻位置不同，只评估观测扰动，不构成相同状态加速试验。第一token仍来自prefill。user时间线包含观测/packetIO，不推广为clean TPS。原始scalar packets留实验端；只发布摘要/路径/SHA。
CPU入口 CPU_inside_r168.sh：实际Native/48objects/DTK/loaderforward AST/冻结问题闭包+T1 schema坏包拒绝，无Torch/CDLL/Runtime/GPU。root发布后提供exact CPU receipt SHA，controller再写root-GO并freshKFD→一次8card n05/40min内2200s→同句柄终态→postempty。没有任何自动重播。
Native已有16个嵌套owner span及6个attention兄弟span。报告各device顺序43层和，各设备不相加；stream elapsed包含HOST launch/dependency/communication等待，不是pure GPU busy。FFN TP_reduction已有单独span，attention output_projection_collective仍融合投影/cast/collective/sync，不能声称纯网络时间；主桶不足时标UNKNOWN不伪造归因。
