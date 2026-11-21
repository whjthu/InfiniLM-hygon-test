# 驻留 BF16 投影的 64 调用图重放先导

固定原 M184、K4096、N1024、BF16，两个输入交替使用，同一个权重与两个预分配输出。eager 每组直接提交 64 次 `torch.mm(..., out=...)`。Graph 臂只捕获同一 callable 一次，随后每组重放一次。两臂共用同一设备、流、缓冲和完成边界，计时内无附加 DMA。Graph 创建和 capture、准备、等待、首个校验单列。

原 640 卡 [小形状 Graph 对照](../../../results/bw1000/2026-11-19/fleet640-small-bf16-graph-original371-198-terminal.json) 在 M1、M16 和 32 个投影的链上实际通过。原 [LoRA 前反向 Graph](../../../results/bw1000/2026-11-19/fleet640-lora-graph-original372-199-terminal-failure.json) 有 capture 错误，因此本 M184、64 次及八 owner 配置有独立 capture 和数值门槛。遇到 capture 或 replay 错误，保留失败臂，不用 eager 代填 Graph 值。

图捕获后在原设备地址上反转两个输入并把两个输出都填为 NaN。重放后检查完整结果及 CPU 前 16 列参考，再恢复输入、置毒及重放。正常输入的完整结果逐 BF16 位比较，CPU 参考按原 BF16 数值检查。改号试验允许数学零的符号差。地址改变、旧输出、缺臂、完整输出失配或原 CPU 参考失配均阻断。每个计时臂开始前再置毒，验证和置毒都在计时窗之外。显存和 capture 成本分列。

八卡先导仍用 O01 `1103606/f12r2n13`，以新鲜 CPU-only 准入为准。预算 12 CPU、64 GiB、8 DCU、140 秒。由同一个原 epoch 下发固定起点和截止，建议 +60 秒起跑、+125 秒 worker 截止。eager、Graph、Graph、eager 四窗相对起点 0、12、24、36 秒，每窗 8 秒，等待不计入。每窗每卡驱动采样 +1、3、5、7 秒，八卡共 128 次有效覆盖。阶段缺失保留，结果读取器从原完成调用数及完整窗墙钟重算有效 GEMM/s，并逐窗匹配真实原 busy 采样。两组配对用于先导判断。

按 `SOURCE_FILES.json` 中键名 staging Python 模块，使用原 DTK/PyTorch 环境下 `python3 -B -u worker.py <私有结果父目录> <本次 plan.json>`。plan 使用 `request`、`campaign_epoch`、`shared_start_epoch`、`deadline_epoch` 和节点的 `selected.{parent,PCI}`。之后由根协调者核对原 Slurm 终态并取得独立零卡清场。私有结果目录用 `python3 -I -S -B report.py <原节点目录>` 只读汇总。当前只提供本机实现与验证，实际 capture、数值、性能和卡利用率均待目标测量。
