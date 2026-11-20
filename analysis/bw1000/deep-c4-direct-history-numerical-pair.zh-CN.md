# DeepSeek C4 直接历史读取的数值对照

原步骤 `1103606.487` 已完成目标 CPU 编译和链接。新 TU31 与 49 个原对象组成候选 DSO。目标算子和真实权重下的结果需由独立设备试验确认。[原编译结果](../../results/bw1000/2026-11-20/deep-c4-direct-history-original487-compile-terminal.json)

`tools/compare_deep_c4_direct_history.py` 接受已完成的基线和候选带权诊断的证据清单。清单固定 `original_range_receipt_SHA256`，并在 `arms.baseline`、`arms.candidate` 各记录 `directory`、`Native_SHA256`、`outcome_SHA256` 和 `progress_SHA256`。目录内读取 `weighted-outcome.json`、`progress.jsonl` 与三条选定 T1 的六份分数或索引文件。运行 `python -B tools/compare_deep_c4_direct_history.py PAIR.json --manifest-sha256 PAIR_SHA` 得到摘要，原数据留在授权位置。

比较器核对同一份原七段权重回执、基线 Native 固定身份、全部 71 个步骤、16 GiB 显存下限、七段加载及逐步输入保护计数。逐一核对投影和缓存诊断的六个采样点。`warm_abort` 和 `primary` 都在 `past=8496`，符合直接使用已收集历史的计划条件。`full_boundary` 从 `past=32767` 追加一个压缩槽，保留原回退路径作为对照。

各样本分别核验私有结果的字节摘要、连续张量定义、有限分数和有效 TopK。分数使用 FP32 字节精确比较。仅当两组分数完全相同、两组 TopK 均各自符合降序分数及范围检查时，才将索引差异判为并列顺序差异。独立运行的整缓存摘要相等性另列，供定位差异。比较器核验的是结果，设备上真实加载的 DSO 身份、算子实际分支和 Slurm 终态仍需原运行证据。

本机 4 项回归通过，覆盖等分并列、全上下文边界分数差异、缺失步骤、原权重回执不匹配。使用完整合成收据，未读取真实权重、发起设备运行或产生性能结论。
