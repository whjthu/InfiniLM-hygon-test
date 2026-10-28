# r168a 两次独立单卡分配probe控制器

原封存r168源码不变，MF `3a7d669d709d13d9282164aba2dbb1081538741c6775e0e06cffee026b1467a3`。绑定实际CPU523 receipt `9cfd80558e16c80d34d617191f41dd51ed8844e1e1a99e1eaa41fb472e09aa63` 及两arm Native/source SHA。只允许root CPU keeper910407控制parent910374、a02r4n01。

部署到 `/refactor/incoming/Kimi-I96-storage-launch-r168a`，root发布后用既有15分钟CPUkeeper执行 `bash CPU_inside_r168a.sh <publicationCommit>`。顺序为owner核验、空KFD、regular独立单卡step完成/marker/别名/Runtimeclose、空KFD、exact新step同样检查、最终空KFD。每GPU step最多4分钟，内层180秒；外层keeper脚本875秒、控制器正常预算840秒。永久唯一intent防自动重提；失败只清理本控制器真实handle，不操作其他作业。

GPU脚本复用锁定CPUreceipt完整LD依赖路径、Common/SDK/DTK，不importTorch、不编译、不初始化CCL；三种VISIBLE masks均设0，仅一张可见GPU。没有IB或CCL资格要求。实际阶段mem_get_info用于afterRuntime归一化的bank占用差；Native pool reserved保持UNKNOWN。

预期regular bank约598MiB、exact逻辑468.5625MiB，设计差129.4375MiB=135725056B。实测差偏离设计超过16MiB或差不大于100MiB，仅标分配结论未闭合，不把执行正常改成Model准入。Runtime背景变化允许16MiB作为诊断范围，不要求raw bytes相等。92层12,486,705,152B是设计外推，不能从58.9GiB再减11.629GiB。无完整Modelfit/性能结论。
