# MiniCPM-V-4.6 Checkpoint 20 原生首前向诊断

同一正式 T05 配方从原 `1103606.1171` 留下的 Checkpoint 20 恢复，八个 rank 各执行第 21 步的第一个微批次前向。验证完成需要八份有同一配置 SHA、实际 Slurm step、起始时间及截止时间的 ready 与 stop 凭证、零反向和零优化器更新。该用例可定位首微批次在 RNG 恢复、Trainer 入口、模型包装、内层前向或 SDPA 的变化。旧运行未采集这些边界，秩 1、6 的原首次差异发生在后续微批次，当前诊断覆盖首微批次。

原 25+5 源循环、八卡 DDP、模型/数据/配置、原训练入口、Checkpoint 20 和 Swift 回调名保持原配方。基于原训练组实物生成新 argv 时只替换观察插件路径、输出目录及本机端口。原 checkpoint 与 framework 在容器只读绑定。HOST 预检只读取原 checkpoint 文件清单并检查原 14 项的规范路径及长度，实际启动时继承的原 callback 负责原 checkpoint 的内容校验与原生恢复。

## 目标入口

把此目录和 `CLAIM.json` 指向的原控制清单、原终态和 observer 清单按仓库相对路径部署到同一个经 root 核准的 checkout。原控制清单中的全部八个成员与 observer 清单全部五个成员由实际 HOST 验证。使用以下命令，其中 `REVIEWED` 指该 checkout，`ADMISSION` 指 root 门禁的小型 JSON 文件。

```bash
HOST="$REVIEWED/code/bw1000/training-minicpm-checkpoint20-first-forward-target-v1/host_once.py"
/usr/local/python3.12/bin/python3 -I -S -B "$HOST" --preflight \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA" --source-sha256 "$SOURCE_SHA"
/usr/local/python3.12/bin/python3 -I -S -B "$HOST" --launch-once \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA" --source-sha256 "$SOURCE_SHA"
/usr/local/python3.12/bin/python3 -I -S -B "$HOST" --status
```

`SOURCE_SHA` 是该目录 `SOURCE_FILES.json` 的精确 SHA256。root 从 `CLAIM.json` 取得 admission 所需的 `request`、`original_step`、`parent`、`node`、`port`、`stage_SHA256`、`checkpoint20_inventory`、`planned_optimizer_updates`，使用 `schema=MINICPM_CHECKPOINT20_FIRST_FORWARD_ADMISSION_V1`，写入实际普通发布的 40 位 `publication`，待外层核实新鲜节点、项目 UID/配额、独占队列预约、现场去重与只读原件后置 `root_GO=true`、`GPU_GO=true`。预检独立于启动，可先收集这组已束定的实际证据。冻结源码自带的 claim 保留未准入状态。

单一新目录在 Slurm 前写 intent。已存在的任一精确标签或目录阻止再次启动。原已验证的 HOST 库控制零卡 pre、八卡 GPU 和零卡 post，同时保留同一 epoch、每步分配和实际记账。常规物理清场验证同一 boot/DRM、端口、无 KFD holder、无同 UID 打开设备的文件描述符及精确 PID/startticks；容器 leased 步额外验证 8 项实际 PCI/KFD 拓扑和 HIP 枚举。已知 GPU 失败仍走清场并保留八 rank 分母。可见 Slurm 终态缺失时留存 UNKNOWN，由 root 针对同一 request 对账，保留独占预约。脚本内 post 是该链的清场，root 释放队列预约前还需要其独立 fresh 清场。

首前向 CPU profiler 和观察钩子有开销。结果标识一次短诊断，训练恢复资格和正式性能字段为 false。组件输出只有状态和比较摘要，原题、原数据、模型和权重均留在目标私有目录。
