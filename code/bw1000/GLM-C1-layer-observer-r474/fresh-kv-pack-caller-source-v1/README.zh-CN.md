# GLM-5.3 fresh K/V 调用接入

此包从 CPU308 的精确 NoV 和 prepare 源码生成独立调用方。按原 1103606.310 的实测轮，fresh.expand_cached 区间含真正 kv_b FP8 投影、K/V 组装和原同步。新调用接收当前那次 FP8 投影得到的 BF16 view，保留 1792 列、相同 provider、相同设备及 Runtime 调度。AttentionPrepare 的声明文件逐字保留，原历史 selected-V 路径也保留。

INFINILM_GLM474_FRESH_KV_PACK 只接受精确的 64 或 128，其他值和未设置都走原拷贝。进程在首次使用时固定该值。NoV 只在本 Runtime 的 fresh 调用配置一个异常安全的作用域，调用一返回便释放。正式门槛为一个请求、一行、已有前缀和四个本地 head。调用点只持有这些元数据，不能仅凭它们判断单行属于真实 AR 还是 prefill 尾块。真实 AR 性能要由之后的调用者单独证明。

命中还要求原有 owner 和 graph 检查、非空 owning stream、同设备且 contiguous BF16、正确形状和逐维元素步幅。投影张量为 [1,4,448] 且步幅 [1792,448,1]，共享旋转张量为 [1,1,64] 且步幅 [64,64,1]。K、V 各 [1,4,256] 且步幅 [1024,256,1]。输入元数据失配沿用原拷贝。候选输出分配异常或 C ABI 返回失败沿原 Work 退出并保留错误，不重做投影或试另一个路径。原观测标记和同步语句保持。

    python -I -S -B render.py <new-source-dir>

渲染器读取 CPU308 PLAN、两个原翻译单元和同一声明。目标接入前需要原 fresh-kv-pack 的真实编译终态、对象 SHA、两个导出和清理证据。独立候选库需核所有受影响翻译单元的 fresh M/actual MD、原 32 对象位置的相对顺序、新 GPU 对象加入位置和 provider。原生产库继续独立保留。性能对照使用同一候选库的 OFF、64、128 和相同原 checkpoint 权重、provider。

实际组件在给出性能结论前，保留三组完整配对和失败分母。每组使用同一 Runtime 配方生成的真实 FP8 投影、共享旋转 K、相同输入及上下文，三个顺序轮转。先核完整 K/V 逐位一致、原输入不变、显存红区、原同步和生命周期，再按同一卡和正常 device 流记录组装区间与完整 fresh 区间，原始延迟逐项保存。

本机编译从生成源码抽取真正 fresh call 与 expand_cached 方法体，用两个 C++ 翻译单元记录输入、原投影指针、调用分支、同步和失败。Runtime、原 FP8 provider 和设备 C ABI 在本机由测试对象记录。
