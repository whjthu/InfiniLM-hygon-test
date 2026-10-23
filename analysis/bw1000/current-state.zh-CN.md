# BW1000 当前性能与后续方向

本页汇总2026-10-24北京时间已取得的结果；运行状态以带时间的独立观测为准。

| 模型/配置 | decode 总吞吐 | 资格及限制 |
|---|---:|---|
| Kimi TP8×PP4，B64/M16，32卡 | 133.59 tok/s | 先前已验收静态512输入/max128输出，约4.175/card；非单用户/在线速度 |
| Kimi TP8×PP4，B16/M4，32卡 | OFF61.305→ON61.857 | N4候选配对约+0.90%，完整token通过，但收益与波动接近 |
| GLM TP8×PP2，B16/M8，16卡 | OFF33.471→ON33.960 | resident K/V约+1.46%；未超过历史同形状34.003点值 |
| GLM TP8×PP2，B32/M16，16卡 | 历史43.50058 | 工作负载/轮次不同，不能与B16直接当代码加速 |
| DeepSeek tailSUM组合 | 不采用 | 第35索引最终token仍不同，纯B32控制与restore通过，QA18通过，ABBA跳过 |

Kimi B16最新first-ready用户均值3.126 tok/s，postanchor约3.877；二者计时窗口不同。在线动态服务应直接依据每请求token交付时间线，单独报告TTFT、平均decode与ITL尾延迟，不能把microbatch数当用户数。

已完成的Runtime诊断显示：预建views等可省约3–4us owner CPU，但多数M1/M4实际完成时间几乎不变；guard/cache-key/allocator微优化不是已证实的数倍加速路径。跨GPU的不同M观测有混杂，不能归因某个backend算法。

GLM真实layer38/真实权重、合成hidden测试约5.36–5.66ms热态wall、3.66–3.84ms owner CPU；gateup约1.09ms、down约.65ms对应低ownerCPU大项，attention prepare与row setup有明显CPU成本。stream事件含host发射空隙/依赖等待，不能把全部当纯GPU时间或PCIe延迟。单层结果不能乘层数作为模型极限。

正在推进：Kimi词表头32卡同次加载B1/B16；DeepSeek五组teacher36诊断区分tile/单行decode/prefill/layout；在线动态调度仅完成CPU控制核心及指标，尚未接Engine/broker。未得到新的15tok/s用户结果、在线goodput、训练草稿投机加速或硬件物理上限结论。

## 词表头整模型终态补充

500已COMPLETED0/40:37，32owner关闭、独立501四节点KFD空。B1配对5.30629→5.34573 tok/s（约+0.74%）；B16配对59.83117→59.86904（约+0.063%），first-ready用户均值3.03590→3.03494。完整对照通过，但未建立明显整模型收益；组件3–5倍不能外推。B1只具同次加载matched reference，不能当独立历史gold；现有总吞吐纪录和15/user未改变。

在线控制原型已修复有界队列/事件与回收、8192 packed-token上限，并通过1000个CPU epoch；完整请求与censored prefix指标分开。实际跨节点broker/Engine适配仍未执行，不列服务性能。
