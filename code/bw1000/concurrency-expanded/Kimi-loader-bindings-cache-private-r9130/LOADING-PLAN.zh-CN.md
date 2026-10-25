# Kimi加载瓶颈：当前证据与下一最小修改

当前407四节点加载分别为1891.65、1526.50、1723.25、1806.02秒。它们已完成权重加载；本次只读分析没有改live源码、SDK、进程或新增GPU任务。

一层mmap约1秒仅建立16.1GB逻辑映射，既没有完整读取全部权重，也没有完成整模型GPU上传。当前fullModel已使用safe_open，权重循环没有.clone/fulltensorSHA；654的clone/hash缺陷不能归因于当前fullModel。

实际开销线索：每个加载事务重新构建state_dict；实际363/378个事务。CAPI的243138/253914是8owner去重union，不能当每owner条目数再无条件乘8。可证明合计map访问范围每节点8826万–7.06亿、9598万–7.68亿；下一真实load需记录各owner实际size。另有afterload累计64.6万/67.5万次CPU非连续copy，23.2亿/24.2亿短block、124.2GB/129.4GB。缺阶段计时，不能把它们直接宣称主要wall瓶颈。

CPU329在7秒完成实际5384key元数据+MockTensor检查。固定64个事务，4pairs基线重建64次约0.099秒，缓存含finalize复核重建2次约0.00327秒。15项身份、线程、shape、strict、重复、partial failstop、late注册/storage变化及清理检查通过。约30倍是纯HOST mockregistry结果，绝不代表整模型加载加速。

先做默认OFF的模块级有界事务：保完整source形状/量化字节、safe_open生命周期、原LoadBatch256和全部posthooks；让一个模块少提交几次。12–17GB逻辑views不代表同等residentRAM，本次原始cgroup探针是v1未解析，须先记录实际headroom、RSS、pagecache。不能直接无条件把2GiB扩大。

再把已验证HOST缓存sidecar接到owner/model/generation明确的Loading生命周期；不改任何类ABI，不缓存source裸指针。finish重新取actualbindings核key和目标Tensor身份/shape/strides/dtype，清strongTensor后再posthooks；所有error/destroy在owner线程Runtime销毁前清理。当前只有CPU模板与fixture，尚未接入真实Model。

InfiniLoader已知非main88d6bac6有raw generation、run lease、完整overlay与有界residency；当前预算不能放下全部rawweights。优先下一模块nodeowned range预取和已验证generation复用；若以后实现rank-major转换缓存，必须新增明确pre-sharded入口，不能将切片再次交旧select_source。后续只在本来需要的新Model加载中测，避免为了loader额外完整重载。
