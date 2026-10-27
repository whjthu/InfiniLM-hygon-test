# Astra 117：预加载源码审核

本轮未发现新的立即预加载阻断项。原 GLM Bash CRLF 已由 root 在远端执行前拦截，并用不可变 r129b 修复；我复核了新 manifest `b2045541…` 的全部 20 个 staged 文件，只有 shell 文件改变。Kimi r127 stage manifest 的 71 个文件全部匹配；CPU449 receipt 所列 `native/` 文件也全部匹配实际 r123e 源码。

这是源码结论，不是 Model 性能或运行 GO。没有 SSH、Git、上传、Slurm、编译、runtime import、模型加载或 GPU 执行。

## 必须区分的结果边界

- **GLM 优化覆盖**：`glm-m15-model-entry-r129b/audit_r129.py:19` 在 M15 eligible/selected/runs 全为零时也能通过，最后 `:38` 仍输出 qualified。天然 EOS workload 未必出现 T15。真实结果必须要求所有 owner 的测量 ON/OFF 确实覆盖 M15，才能讨论该优化是否有效；可在只读结果汇总处加门槛，无须改动本轮冻结 workload。
- **Kimi 完整覆盖**：`Kimi-ctx2k-Model-controller-private-r127/launcher/model_audit.py:28` 接受任意 truthy `resident_stop`，包括正常 q9999 stop，因而不能仅凭 `passed` 宣称 18 case 全部完成。请同时看 `initial18_complete`、缺失 ID、`command_rejections`、quality screen 与 budget ledger。当前 `passed` 只能当生命周期结论。

## 已核查的关键逻辑

- Kimi runtime/loader/bindings 路径来自 manifest；create_ctx2048 CAPI 指针类型和 symbol provider 检查完整。`runtime/target_model.py:51–56` 在读权重前核对 PP 分区、真实缓存几何和全部 93 层 key 覆盖。
- r123e `native/pair_probe.cpp:55,283,300` 将 ctx2048 限于显式 env、TP16 PP2 logical8；reset 延续真实 2048 上限。`native/kimi_compact_allocate_r4974.cpp:74–76` 保持 physical16、132 pages、17 KDA rows，未把 KDA pool 错扩成 33。
- `native/worker.cpp:1150,1166,1238` 在 owner reset/shutdown 清除状态并同步；分配异常在 backing 仍存活时先 drain。`runtime/parameter_loader_r9143.py:19–34` 保持 CPU shard、A_log trim、2 GiB flush、同步 load 后才释放源 Tensor。
- GLM `model_entry_r129.py:210–217,238` 检查 17 stats、defaultOFF、非法长度、错误线程无污染和析构后零 active bytes；`glm_fp8_runtime_oproj_r125.cpp:56–68,91–99` 只选精确 M15 o_proj，按 owner 清缓存并先 drain。
- Kimi nonce 不写入 launcher 原始 argv/report；本次查看的超时调用不携带 nonce，握手不回显 nonce。清理以记录的实际 handle 为界。没有发现新增 GPU F64 操作；CPU 时钟/统计 double 不等于 GPU F64。
- 队列为 2 warm + 8 C2 observer + 8 MMLU，MMLU 输出 cap1024、prompt+cap≤2048、无输入截断；固定首 8 题配对质量 screen 保留，不以 token 一致作为通过条件。

## 后续修正

- `runtime/model-manifest-r9183.json:160` 的 planned_case_count 仍为 10，实际队列为 18；这是描述性陈旧字段。
- GLM `audit_r129.py:4` scope 仍遗漏 M15 treatment，应在发布摘要明确 M15 ABBA、row2 两臂固定 ON。
- GLM `glm_arrival_probe_r129.py:67` 每 16 个非 warm command 更新心跳，而 `controller_r129.py:48` 用 61 秒判断 stall。16 次成功但较慢的调用也可能触发误取消；本轮尚无这种运行证据，后续应按调用返回更新或分开单次调用 deadline 与 episode 心跳。

## 最有价值的下一步

先用本轮同一冷加载的真实数据决定优化方向。GLM 先看 T15 实际频率、native wall 占比及缓存 hit/miss/eviction，再比较 ABBA 的端到端指标和质量，不能把组件 4.47–4.50 倍外推到整模型。Kimi 用固定 C2 observer OFF/ON 测观察开销，并用 `phase_observer_r126.py:47` 的本地 interval union 定位 verify/replay、draft RPC、memory/control 的主要耗时；嵌套 span 不相加，也不能称 GPU busy。

完整逐项路径、行号、哈希及局限见同名 JSON。第三个模型、protected 910384 和未展开的继承远端 helper/binary 不在本次结论范围。
