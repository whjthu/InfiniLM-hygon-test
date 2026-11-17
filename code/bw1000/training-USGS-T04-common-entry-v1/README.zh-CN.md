# 统一训练入口的 USGS 显式分支

只在 T04.sft --usgs-domain-config PATH --usgs-domain-config-sha256 HEX 时进入新领域配方。其余17项默认入口、原396安装与在途作业保持原行为。

此包只有两个入口源码 overlay 和新的纯CPU路由检查。与三个 domain observer 模块合并后是400文件共同框架，不复制训练循环。Fresh25 与恢复20到25均仍由原生 Swift/HF 执行；领域与通用对照是独立配方，不继承正式性能资格。

启动前核真实配置、完整400源码、原生45输入、800行曝光身份和已登记provider；随后单次核8张A100物理身份/进程/端口，再做CUDA runtime probe。启动前必须在同一进程monotonic时钟确认原准入不超过60秒，不重新采样重置。

当前只有源码。GPU_GO=false，尚无实际安装、模型或训练。入口内部subprocess超时不能证明8个训练后代已清理，未来具体GPU包必须真正包含原成功的held-Popen、PID/startticks登记、全树TERM/KILL/reap和独立post，才可启动。原UNKNOWN范围不能借此重播。

新CPU检查：3种有效计划、10种坏计划拒绝、6种分派路径，以及大于旧900秒的显式remaining绑定fixture通过；无Torch/SDK/model/GPU。首次测试异常类型不匹配和源码拼接缩进错误保留在SOURCE_PROOF，均未触远端。原测试不重跑。
