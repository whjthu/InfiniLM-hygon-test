# MiniCPM Checkpoint 20 RNG 三臂目标控制器

在同一合格八卡 Slurm step 依序启动 `a1 → b → a2` 三个独立 Swift/torchrun 进程组。从 `.1171` 留下的同一 Checkpoint 20 原生恢复，每组保全八 rank，每 rank 一次原首微批次前向。新诊断沿用已经真实走通的 `.1189` Slurm、物理快照、容器与原语，并固定 `.1191` 独立清场正本的 SHA 做历史证据。目标仍需要 fresh 准入及本次独立清场。

本试验保留三次真实模型初始化。单一 HOST epoch 的各臂硬截止为 +460/+840/+1220 秒，启动每臂前至少剩 180 秒。每臂可在其截止后用最多 30 秒完成其强所有者清场。共同 native 见证 deadline 为 +1260 秒，容器 worker 清理 +1360 秒、GPU 客户端 +1440 秒、全控制链 +1800 秒。前一臂的实际返回码、八 rank 完成凭证和正常进程退出都成立，才允许后臂启动。失败和未启动的臂保留在 24 次计划前向分母。短诊断不产生反向、更新或正式恢复结论。

## 根调度器入口

将此目录、`bracket_contract.py` 固定的 first-forward target manifest 及 RNG source manifest 各完整目录、已发布 `.1189` route 和 terminal 文件、旧 first-forward target 使用的原 controller manifest 与八个成员、原 first-forward observer manifest 及各成员、原 `.1171` terminal 保持仓库相对路径安装到同一经审核 checkout。HOST 启动时依据清单核字节。GPU 容器阶段所需 15 个文件由 HOST 安装到同一个新 target 根，旧模型、checkpoint、框架及数据挂载保留只读。

从 `CLAIM.json` 复制 `request`、`parent`、`node`、`port`、`original_gpu_step`、`original_post_step`、`original_terminal_SHA256`、`stage_SHA256`、`checkpoint20_inventory`、`planned_forwards`、`planned_optimizer_updates`，写实际普通发布 40 hex 的 `publication` 和 `schema=MINICPM_RNG_BRACKET_ADMISSION_V1`。现场 project UID、T05 节点独占、fresh PCI/KFD/PID、配额、端口和去重检查完成后由 root 置 `root_GO=true`、`GPU_GO=true`。清单是源码冻结状态。

```bash
HOST="$REVIEWED/code/bw1000/training-minicpm-checkpoint20-rng-bracket-target-v1/host_once.py"
/usr/local/python3.12/bin/python3 -I -S -B "$HOST" --preflight \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA" --source-sha256 "$SOURCE_SHA"
/usr/local/python3.12/bin/python3 -I -S -B "$HOST" --launch-once \
  --admission "$ADMISSION" --admission-sha256 "$ADMISSION_SHA" --source-sha256 "$SOURCE_SHA"
/usr/local/python3.12/bin/python3 -I -S -B "$HOST" --status
```

`SOURCE_SHA` 为本目录 `SOURCE_FILES.json` 的 SHA256。只读 preflight 检查旧 checkpoint 清单及其 14 项成员元数据，另验三臂 Swift argv 和现场 Slurm label/成员。配置只包含原 checkpoint 精确身份，不在 HOST 读原权重正文。新请求目录和 intent 在 Slurm 之前独占建立，已有任一标签或新目录时禁止再次提交。Slurm 终态未知时无清场释放结论，root 只针对该请求读取原账单和现有文件。

新输出根下各臂 `<arm>/train` 与 24 份 rank 证据留目标私有。只有原有 GPU 步终态可见才尝试正常的零卡 post；失败 GPU 也保留 post 和已尝试臂的精确 PID/startticks。HOST 将本次 post 与本次 pre 对比，另外检查每个存在 intent 的臂都有清场回证。释放共享预约仍须 root 单次独立 fresh post。
