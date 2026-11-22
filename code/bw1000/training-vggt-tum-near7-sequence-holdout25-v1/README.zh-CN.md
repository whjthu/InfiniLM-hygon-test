新领域配方：7 对 near 录制训练 25 次真实 optimizer 更新；far 和 sitting 两录制仅做 base/新 camera head 各一次预测。初值是同一已核验公开 base，绝不加载 .927 已训练 head、旧 8-pair UNKNOWN 或 Gemma scope。7 对只来自一个录制，25 次重复曝光不是 25 个独立样本。

这是本候选的按完整录制留出修正。far/sitting 曾被旧 .927 训练，不能称全局未见盲测；本包不打开家庭场景盲测，也不声称大规模泛化。全部 4 个预测先 fsync 封存，随后才由原冻结 pose scorer 读取两个位姿真值，保留失败/退化的全部 2 对分母，输出 AUC@5/15/30 与失败数。指标提升不是运行成功门槛，也不据此晋级模型。

直接运行入口：`launch.ship(publication, fresh_snapshot, root_authenticated_transport)`。它沿根的既有队列/新鲜资源收据注册固定请求并使用根注入的 held transport，无默认 SSH、无 detached 后台 controller。源 flags 初始 false，根审核后只修改 CLAIM 授权并重封 SOURCE_FILES。实际运行前根必须确认 f13r4n09 当前清场、队列无保留、父分配仍合法；历史 .929 clean 不当成 fresh admission。固定请求 be7757b8c88a452a9b6d63d291dd4fb8，不允许重播。

whole 1800 秒包含本机源校验/上传；pre/post 各最多45秒，GPU Slurm 最多1440秒，容器1300秒，worker继承原epoch，留 accounting/post 预算。1 DCU/4 CPU/64GiB。训练只保原 camera-head/BF16、224²双帧、seed20260911、恒定LR2e-5、warmup5+测量20、累积1和 checkpoints10/20/25；原762fc0…数学 backend 不改。既有398源码/74数据文件仍在 .922 candidate，只读复核；新 collector 仅通过精确单文件 :ro bind，不更改 sealed inode/ctime。不重新生成数据或下载模型。

目标路径详见 CLAIM。相同 target 数据/权重已在 .922/.927 实测，但新 seven-pair collector、held transport/metric 一体链仅本机源码和边界验证，尚无目标执行；READY=false，Model/formal17/performance=false。独立终态调用 `read_terminal.py ORIGINAL --actual-accounting JSON --independent-post JSON`，必须取得本次 pre/GPU/post 原行及后续独立清场才能完整 qualified；失败保原更新前缀与全部计划分母。

来源：TUM RGB-D SLAM Dataset and Benchmark，CC BY 4.0（原发布方明示例外保留）；Sturm、Engelhard、Endres、Burgard、Cremers。官方：https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license 。归属沿已冻结公开证据 training-vggt-tum-cross-camera-official-RGB-calibration-controls.json；本配方是派生裁剪/224 resize、坐标转换与训练选择，不是官方 CO3D 配方或官方 TUM benchmark 等价复现。图片、位姿、权重和预测均仅留本机/目标私有目录。
