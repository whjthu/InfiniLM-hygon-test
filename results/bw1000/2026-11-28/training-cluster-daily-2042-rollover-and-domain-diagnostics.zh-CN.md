# 2026-11-28 晚间补充日报

北京时间 20:42。此稿补充 18:01 日报的业务终态和保底续接。逐卡驱动指标沿用已完成的 640 卡完整小时测量。

| 项目 | 已核实 |
|---|---:|
| 分配资源 | 80 台、640 卡 |
| 当前编号步骤覆盖 | 592/640 |
| 本次集中到期后重新续接 | 56 台、448 卡 |
| 完整小时全池真实驱动忙度 | 93.21% |
| 可立即提交的真实训练储备 | 0/5 |
| 正式训练用例 | 0/17 |
| 正式在线推理 | 0/3 |
| 正式领域模型底座覆盖 | 0/12 |
| 正式领域模型交付 | 0/50 |

集中到期曾降至 144 卡有编号步骤。两批在实际零卡物理核验后续接，新增 56 个持有八卡的编号作业。还有六台无加速卡步骤，包含一个业务验证节点、一个端口占用、一个已有外部设备句柄和三个遗留超时问题。小时驱动测量无缺样，距 95% 还差 1.79 个百分点，原测量没有可复核的业务／保底分摊卡时。 [作业覆盖原件](training80-2040-two-waves-expiry-144-to-592-numbered-and-port-veto.json) · [完整小时驱动终态](training80-0445-640-card-60min-driver-busy-93p2065-independent-terminal.json)

- **LLaVA** 两臂 32 槽按原 128 token 全部截断。针对同一终止链路的 0 GRES 检查表明，已曝光控制样本的 tokenizer、模型生成配置、Swift 和 HF 终止条件一致，行为符合预期。保留屏幕的原失败分母，后续按独立任务评估方向，不重复同一盲测。 [GPU 实际终态](specialized-llava-val16-v4-5232-two-arms-all-cap-and-clean.json) · [停止条件](specialized-llava-5236-original-stop-criteria-consistent-before-forward-zero-gres.json)
- **SDXL** 新 CP25 可从本次训后连续和恢复两份检查点读到同一适配器。第一次六图生成止于 Euler 摘要，全部六槽保留缺失。零卡实测中只换默认项顺序即能严格复原原配置，两个进程的原生时间步与旧六槽逐一一致。下次仍按同噪声和旧时间步执行。 [训练恢复终态](specialized-sdxl-2769-four-native-rows-25-plus-5-exact-CP20-and-clean.json) · [Euler 终态](specialized-sdxl-2803-euler-default-order-old-wire-reconstructed-six-timesteps.json)
- **Gemma** 地震图表的独立收益维持 +12.5 个百分点。原 CP20 的 14 个文件大小和引用的 Trainer 身份核实。已准备只观测首次恢复前向的诊断，下一次仍需专用启动和八卡准入。 [CP20 核验](specialized-gemma-851-cp20-fourteen-extents-and-current-trainer-bound.json)
