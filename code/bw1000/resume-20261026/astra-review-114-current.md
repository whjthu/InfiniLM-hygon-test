# Astra 114 · 冻结结论

冻结于 2026-10-26 21:38:36 UTC，10 分钟内完成；10 个输入的实际 SHA256 全部匹配。本次只读审查，无 SSH、GPU、Slurm、Git 或新 agent。详细证据与实验卡见同名 JSON。

- **先落实 113：**K113A 已交付 C2 数据但未执行，原 Kimi 在第 34 个完成 case 后因 MMLU reserve guard 停止；K113B 的 INT8 v1/v2 已完成；G113A 已完成整模型；D113A 已保留负任务 wall 与 opt-in 范围。
- **只消费三轮：**910375.753、910376.755、910375.757。Kimi 470 FAILED/audit1、34 个子集、CPU 准备、未运行 C2 和旧 Deep64 均不计新合格性能轮。
- **Kimi 优先：**保留 4 GiB floor，支持新 immutable cap8 + prefill micro1；同一次必要 load 先完成固定 32 GSM 的 C2 配对，再 C8 与 MMLU31。cap2 被实际 CAPI 拒绝。tail 静态节省 1.02 GiB 尚非实测；两臂同几何，因此只隔离 W2 开关收益，不分离 cap 与 micro 的内存因果。
- **立即修既有接线：**r9293 的 screen_gate 只认 category=screen / 16 个 C1 case，新 queue 却只有 main-C2 等类别，first8 停止条件不会触发。将原规则接到 main-C2 前四组完整配对的 8 个固定 ID，沿用“候选额外 ≥2 错题或 nonEOS/cap”，无需再跑 C1 或增加门禁。已通知 root，冻结时尚未亲验修复。
- **内存失败口径：**T568=159+409 prefill CAPI 成功后，free=4,102,029,312 B，距 floor 少 184 MiB；未发现 OOM。32 owner shutdown 与 post-empty 成立，但不改 FAILED。完成子集 GSM64 decode +5.78%、frontend C32 含队列/PF 服务速率 +3.62% 仍为 provisional。
- **最短判定与成本：**两个短 warm 后先得到首 C2 pair；前 8 题仅作早停筛查，完整 32 题才是本轮 C2 benchmark。计划 load 30–35 分钟，ready 后首 pair 1–3 分钟、完整 C2 15–35 分钟，均非承诺。原 50-case 单 load 队列可保留；不得重跑 64/C16。MMLU 原 32 中 11704 因 576+512>1024 共同拒绝，覆盖 31/32，禁止裁题替换。沿全局 deadline 保留 budgetskip。
- **GLM 已测小收益：**16 卡 task throughput +1.786%，task wall -1.755%，HOST decode +2.477%；两对任务速率均正。32 题各臂 31 正确、32 EOS；用户中位仅 1.054→1.077 tok/s，>15 为 0。row2 实际命中，不能把组件 10% 称为整模型 10%；保留限定 opt-in，本轮零新 GPU 卡。
- **INT8 停止参数救援：**两版高 reuse T64 +28.6%/+28.8%，T8 -4.8%/-5.7%，构造低 reuse 约 -52.8%（延迟增加约 112%），均无 Model 质量证据。87→61 VGPR 未建立整链改善；v1/v2 跨 run 不构成直接因果配对。
- **最多一次诊断：**Kimi 第二卡仅允许 bounded phase events + 实际 launch 元数据，问清低 reuse 多出的 8.23 ms 落在 quant/pack、matrix、reduce 或 W13/TP 哪一段。保留原三 scenario/基线与整链 control，不扫 grid，不在段间同步，不求和各 rank 段最大值冒充关键路径。753/757 已结束，不能称仍驻留；严格无新增 load 时只能搭载下次本来必要的组件驻留，禁止为此冷 load 完整 Model。不能归因则 UNKNOWN/PARK。
- **静态猜想须修正：**假设 wave64 与 80 CU，640×64 仅等于平均 8 个 launch waves/CU，不是实测 occupancy，也不能推出降低 VGPR 绝不会增加 resident waves。物理 SKU 峰未知，大矩阵 INT8 TOPS 不是模型屋顶。
- **Deep 零新卡：**复用已审 64 题与 n05 task wall 回退 3.12% 的结果；C4 opt-in，MTP 配置路线 STOP，不重复 64 或展开大实现。

下一步共两张 Kimi 卡，按“C2 单 load 完整判定 → 可选组件归因”排序；GLM、Deep 均零张。所有质量判断允许 token 不同，无 GPU F64，不以小样本无差异宣称正式非劣。

