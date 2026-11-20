# Qwen USGS 80 题开发集原运行验收

原请求为 `bb61591025d94fdaba421ac15d5d4fa7`。集群只读对账在 SSH 用户认证阶段被拒，原运行目录和 Slurm 状态仍待核实。[原认证记录](../../results/bw1000/2026-11-20/training-qwen-usgs-dev80-pre-exec-auth-rejected.json)

集群连接恢复后，先检查原请求。双臂全部结束后，在集群上使用 `tools/read_qwen_usgs_dev80.py capture --output <新私有目录>` 采集原件。该命令从 Slurm 独立核对底座与适配器各自的预检、计算、清理步骤，再复制到独立目录。目录内含原始预测，留在项目私有存储。

将采集目录带回本地，用 `tools/read_qwen_usgs_dev80.py verify --bundle <采集目录> --dev-inputs <原开发集 swift.jsonl> --gold <本地 labels-private.jsonl>` 验证。验收要求两臂均完成 80 题、题序一致、原始预测和计分投影一致、每臂实际一张卡、清理前后设备与启动身份一致。输出保留截断与解析失败，按 20 张源表列出同题得失；此结果用于开发集诊断。

本地验证读入了 80 条冻结的开发题及其私有答案，配对验证分支用合成终态收据执行。通过完整分母检查，拒绝缺失 Slurm 行、损坏预测、损坏进度和错误请求四类证据。本轮实际模型生成数为 0。 [独立验收工具](../../tools/read_qwen_usgs_dev80.py)
