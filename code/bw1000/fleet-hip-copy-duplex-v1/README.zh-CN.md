# 节点内八卡 HIP 双向搬运对照

同一物理卡使用两条 nonblocking HIP 流和两套独立 endpoint，比较 8 次 H2D 全部完成后提交 8 次 D2H，以及两批搬运提交后再统一等待。两臂保留相同内存类型、每批字节、方向顺序、两个流和两个事件区间，唯一变量是提交 D2H 前是否等待 H2D。测试 1 MiB、32 MiB，每个大小按 serial/overlap/overlap/serial 执行，每臂 3 组预热、12 组计时。每个方向使用 `hipHostMalloc` 固定页内存，搬运以实际 `hipMemcpyAsync` 进入目标 provider。文件系统、权重、Torch、网络及算力 kernel 不进入窗口。

已实测的历史 `910374.707` 使用热 mmap、单方向 H2D 和运行时 flag 单变量，覆盖一台八卡。本任务回答实际双向 DMA 同时占用时的单卡和节点带宽，供推理加载、卸载或训练 checkpoint 的并发预算参考。来源见 [已封存 707](../../../evidence/bw1000/pageable-H2D-qualified-r8420.json)。

每个节点的 8 个 owner 在同一轮次屏障释放，各自记录本机单调墙钟开始/完成、方向流事件时长。node 容量按 8 卡总 payload 字节除以本机最早开始到最晚完成的时间；跨节点单调时钟不拼接。预热之后污染目标缓冲，随后每臂重读两方向完整 payload、完整原输入和所有 guard。真实 API 成功和精确字节是组件证据。每臂的原方向耗时和共享墙钟都保留。计算利用率由另一次现成同步 sysfs 采样报告，DMA 完成和显存占用与计算 busy 分列。

## 入口和资源

冻结目录中的 `SOURCE_FILES.json` 指向真实需要复制到 worker 同目录的三个已审模块及本次 worker；入口会在导入前重新核字节。使用现有 `dtk26-sandbox-r1`、`/opt/dtk/env.sh`、原 provider 路径与 SHA。无需新编译或安装运行时。

```sh
python3 -I -S -B -u worker.py /private/unique-run/nodes <original_epoch_plus_180> /private/unique-run/admitted-pci.json
```

第三项是既有准入 receipt 选择的 8 PCI 全映射，以节点短名为键。每节点 1 Slurm task、12 CPU、12 GiB、8 GRES，每个 owner 固定一个不同的 CPU，优先选已分配 CPU 中 PCI 的 `local_cpulist`。报告实际 CPU 和 PCI NUMA 归属，页的实际驻留未知。逻辑同时分配上限每卡约 64 MiB pinned +64 MiB device，另有验证临时内存，实际 driver 占用从采样记录。初步预算为 Slurm 2 分 30 秒、同一原 epoch +180 秒为 worker 绝对截止，HOST 对账窗口需另留余量。已开始的发送、加载和重试须使用同一原 epoch。提交前确认节点当前空闲和本组训练优先；只有认领明确允许的借用组才入队，按单父作业和单出借组分片。满配 80 节点 / 640 卡是可覆盖上限。受现有训练、在途/未知请求、保护节点及 fresh PCI/KFD/PID 准入限制。

收到 SIGUSR1 后在下一个完整搬运组边界让出，保留已写的逐臂 journal，并在流同步后归还。硬截止强制退出，按原 Slurm 终态和独立节点清场确认释放。节点提供 `intent.json`、八个 card JSON 以及 `terminal.json`。任何失败、缺页、缺方向、PCI 不一致或未正常释放按原计划分母计入。协调层仍需完成认领、现有准入、终态/独立清场及会计收据。本机尚无实际 HIP 设备执行。
