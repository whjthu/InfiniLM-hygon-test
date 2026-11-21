# 显存驻留 BF16 投影与双向 DMA 争用

计算沿用 [原 Sync-depth 实测](../../../results/bw1000/2026-11-22/fleet608-qwen-projection-sync-depth-terminal.json) 的 M184、K4096、N1024 和同一 BF16 `torch.mm(inputs[j], weight, out=outputs[j])`。两个固定驻留输入、同一权重和预分配输出，每组 64 次。原 Sync-depth 的 32 个分片均已终结。本候选比较同一个投影在有无双向搬运负载时的速度、数据完整性及设备 busy。

四臂按 compute、compute+DMA、compute+DMA、compute 顺序执行。两臂都持有同样的 DMA 缓冲，两条独立固定页 HIP 流只在加压臂使用。每组投影加压时先向两个搬运流各提交 8 次 32 MiB，再向计算流提交投影。每组记录完成的实际工作量、完整组墙钟、计算流 drain 前后的主机墙钟及各流 HIP 事件区间。事件区间包含该流排队和主机提交的影响，三条流的区间不能相加为 GPU busy。每臂核两组完整 BF16 输出和前 16 列 CPU 参考，搬运臂另核完整原字节和 guard。

最小实机验证先用 O01 一台八卡、12 CPU、16 GiB，先导节点建议 `1103606/f12r2n13`，以新鲜 CPU-only 准入和任务保护为准。使用已通过的 DTK 26 sandbox 与原 Torch 包环境。把 `SOURCE_FILES.json` 中的 Python 成员按键名置于 worker 同目录，`duplex_worker.py` 保持原先导 v2 字节。在原 `/opt/dtk/env.sh` 下运行 `python3 -B -u worker.py <私有结果父目录> <本次 plan.json>`。plan 含唯一 `request`、`campaign_epoch`、`shared_start_epoch`、`deadline_epoch` 和以节点短名为键的 `selected`，每项给实际 `parent` 和 8 个 `PCI`。独立终态和零卡清场后归还。该目录当前提供 worker 和本机流程证据，目标派发和受保护资源绑定仍由根协调者完成。

多组派发复用原父作业及组边界，给所有选定节点同一个绝对起点。建议 `shared_start_epoch = campaign_epoch + 60s`，`deadline_epoch = campaign_epoch + 125s`。四臂起点相对为 0、12、24、36 秒，每臂实际交付 8 秒考察窗口，准备、等待及末尾核验单列。准点错过超过 500ms 时该臂留缺失分母。每臂在 +1、3、5、7 秒另采全部八卡 sysfs，保留原采样 epoch 和缺失。总卡峰值以 GPU 步骤的实际起止单列。时钟校准不足时只报实测各节点时间及记录覆盖，`guaranteed_overlap` 需要每台独立钟差上界才能给出保守同步重叠。扩展仍以逐臂实际运行、数值、终态和清场为准。
