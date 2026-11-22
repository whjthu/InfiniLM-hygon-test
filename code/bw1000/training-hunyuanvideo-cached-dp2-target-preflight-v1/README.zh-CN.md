# HunyuanVideo 原 DP2 配方目标预检

本项是 T09 双卡数据并行。两进程、SP=1，复用 .794 已封的原 480p Transformer、原 Muon LoRA 和 3,521,630 字节条件缓存。原 A100 DP2 配方指定同一个缓存，两 rank 用各自的 RNG。原连续窗口为 5 步预热加 20 步正式计时，分别在 10、20、25 步提交双 rank 检查点。原 .794、.828、.884 是单卡证据。

先从已有授权的 A100-27 原目录绑定 `node/results/hunyuan-cuda-dp2-baseline-r1-repeat1/launch.json` 的 1563B 原字节。公开摘要缺少配置值，不能用它重建原 launch。固定 SHA256 为 `181b1ca706a7d62741298192c3391a0336acf0c60624aa9f2645e90a55934a6f`。

发布后在目标运行 `preflight.py host --a100-launch <原文件绝对路径>`，读取原有目标资产和 393 文件基线清单，重核共同 CLI 与两个 DP2 底稿。使用预检输出中的三个只读绑定和既有训练容器进入无卡模式，再执行 `preflight.py mounted --a100-launch <原文件> --asset-map <host 返回的 asset_map 原样对象>`。该路径运行共同 CLI 的计划模式，只验证目前实际安装的 DP2 overlay、unpad 和配方。两份 DP2 文件为已审本地集成的原字节。原目标权重仅核元数据并重用已封的整文件校验，实际运行仍沿原目标加载边界验真。

两阶段通过后，根再从 `1087656/f11r4n17` 获取同一步新鲜 PCI/KFD/PID 及两张独立卡分配。采用同一双卡生命周期和有界控制器，分别由新的两进程组执行 continuous、checkpoint-10 恢复和 checkpoint-20 恢复。恢复入口使用本次 continuous 的 `COMMITTED.json`，校验两个 rank 的状态和设备。`native_argv` 生成同一共同框架调用。读者沿 `training-hunyuanvideo-dp2-evidence-v1/audit.py` 复算实际两 rank 和最慢窗口。正式三轮及其他验收仍需对应真实结果。

当前本地只执行源码与配方边界检查。原 launch 的目标字节、两阶段目标预检、本次两卡准入和真实训练均待完成。
