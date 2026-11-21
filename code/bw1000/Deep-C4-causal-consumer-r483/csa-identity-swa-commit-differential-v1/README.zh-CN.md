# Deep CSA TU07 提交分支数值对照

两臂使用同一个已独立对账的 TU07 候选 DSO，在同一物理卡上用两个独立进程执行。唯一开关是 `INFINILM_DEEP_CSA_IDENTITY_SWA_COMMIT`，OFF 为 `0`，ON 为 `1`。固定 layer 2、两条状态行、32768 上下文、128 滑窗、BF16 隐藏宽度 4096。由同一输入计划预先固定实际权重、隐藏输入、位置及初始状态。按原 `CsaAttention::forward` 顺序推进到完整 32768；捕获真实 T1、混合请求和窗口环绕、T129 裁剪回退、回退后的 T1 及最大 past T1。

`compare.py --plan` 输出完整预注册调用序列及 SHA-256。准备好真实 GPU 捕获后，在原私有路径运行下列命令。编译对账输入为现有控制器 `--reconcile-existing` 的原始 JSON 输出。

```sh
python -I -S -B compare.py --compile-reconcile /private/TU07-reconcile.json --off /private/OFF --on /private/ON
```

每臂 `capture.json` 含 `compile_reconcile_SHA256`、`library_SHA256`、实际 `CsaAttention::forward` 解析方 `resolved_forward_provider_SHA256`、`plan_SHA256`、`opt_in`、`node`、`PCI`、`slurm_step`、`pid`、`pid_startticks`、`completed`、`normal_destroy`。身份 SHA-256 字段为 `checkpoint`、`runtime_config`、`input_plan`、`capture_driver`、`other_environment`。最后一项规范化时只移除该环境开关。每步保存 `name`、`requests`、原输入字节合并的 `input_SHA256`、`passed`。过程失败仍保留已完成步骤，结果保留两条臂及全计划分母。

五个捕获步另记录 `commit_IndexSelect_calls`、`commit_callsite_trace_SHA256` 和 `raw`。调用数来自真实提交 callsite 的 trace，需在私有结果中保留摘要所指向的原 trace。按固定请求和源码推断的调用数只能作为预期值。每步的 `raw` 是 `swa`、`compressed`、`compressor_state`、`index_cache`、`indexer_state`、`selection`、`output` 各自的 `bytes` 与 `SHA256`。原 `CsaAttention` 的公开只读 getter 可提取前六项。逐个写入同目录的 `NNN-字段.bin`，步号以零开始，`selection` 按请求顺序串接固定小端 I64 张量字节。原 SWA 的 block 256 对两条状态行分配一页，程序核对完整 149760 字节，包括 untouched 区域和页尾填充。对照器重新读取原 bytes 并要求各列精确一致。

这份包提供数值计划和 CPU 证据读取入口。实际设备捕获器、原权重闭包、GPU claim、终态与独立清场仍需交付。正式性能使用独立无捕获测量以及至少三个完整配对。禁止将 trace、权重或原激活放入 Git。
