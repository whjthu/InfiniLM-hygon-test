# GLM474 fresh K/V 两种 tile 的设备事件对照

复用 GPU 步 `1103606.941` 已过数值和参数保护的 .918 Native 动态库，独立清场证据为 `1103606.943`。在同一张 DCU 上直接调用实际 t64 和 t128 出口。T1 采用 TP16 decode 布局。T512 验证 C ABI 支持的最大行数和 prefill 规模的拷贝压力。生产上游的 AR 选择器只启用 T1。

每个形状做三组交叉配对，顺序为 64/128、128/64、64/128。每臂先预热 32 次，再在同一 HIP 流事件窗内固定运行 T1 1024 次或 T512 128 次。保持相同的实存输入、缓冲区地址、BF16 字节、设备选择和固定 8 GiB 余量。每个窗口前后读实际显存，两臂的可用字节前后均匹配才进入配对。预热和计时各自检查完整 K/V 位级输出、输入不变性，以及设备前后红区。失败保留已完成和部分入队计数。

设备事件窗包含实际 kernel 与可能的宿主供给空隙，另报宿主提交、等待和完整时间。输入上传、结果回读及 CPU 检查在计时窗外。每个形状单独报告 t128/t64 吞吐配对。整模型精度、真实 prefill、用户 decode 速度和模型晋级分别验收。

新范围 `GLM474-fresh-kv-pack-event-paired-GPU-v1`，请求 `7c6ec3205bdf41578b0a6d1698787f13`。在根侧新鲜分配验证后，按 f12r2n13、父作业 1103606、4 CPU、16 GiB、1 DCU 启动。HOST 在同一 300 秒窗口完成零卡检查、单卡执行和同节点后验。正式入口如下。

```text
host_once.py --checkout <canonical R/incoming reviewed checkout> --publication <40hex> --admission-sha256 <64hex>
```

目标使用清单里的 `CLAIM.json`、`host_once.py`、`probe.py`、`device_inside.sh`、`timing.py` 和 `result.py`，同时保留 v2 数值阶段已验证的 PLAN、ENV、.918 归档、HIP/KFD 和 HOST/Slurm 辅助源码。另加入 .941 的结果归档，路径和 SHA 见 claim。独立完成读取在本机取原运行目录的精确文件、原始七列 Slurm 记账、独立新鲜的目标节点后验和原队列 handle。它输出唯一 finish 包，由根协调者写入队列。

```text
finish.py --run <original controller result directory> --actual-accounting <7-column PSV> --independent-post <fresh collector JSON> --handle <original queue handle JSON> --out <exclusive output path> --manifest SOURCE_FILES.json --manifest-sha256 <64hex>
```

`python -I -S -B check_local.py -v` 核对 T512 边界、设备事件调用顺序、同显存条件、原分母及独立清场。实际设备事件与吞吐待目标实证。
