# Astra 第110轮审核

已核验原输入 SHA256 与全部10个 sources；本轮仅消费 1102、1108、730 三波。Kimi 466/580 后续经独立CPU422原checks审计与post清理闭合，作为下一轮pending wave1，不混入本次consume3。

- **Deep：继续已授权的小T整Model配对。** 1102 bank wall -3.58% 未达预定5%门槛，维持OFF。1108 T2/T4/T7 单CSA下降21.42%/32.78%/34.61%，足够支持原成功47对象复用、仅改unit19的 C1/C2/C4验证；不乘层数推算收益。正式cap512，cap2仅预热；首配对ready后约5–10分钟。
- **GLM：停止chunk32。** 吞吐 -18.84%、TTFT均值 +47.35%，虽ITL P99改善23.44%，整体不可采用。先闭合当前 a549 最终对象/source 链，再用已有counter和T13/T15/T16小组件配对确认真实命中。r8861只放行routes32/64/128，r8528形式支持其他8倍routes仅是可试验线索。组件正向后才一次16卡整Model验证；首组件结果约25–50分钟（排队另计）。
- **Kimi：epsilon.10不采用，原作业按stop清理。** 新增16case中，token/s +3.31%伴随tokens +3.83%，wall +0.50%；两臂均6/8正确自然EOS、2cap，正确EOS任务/此decode时间略降。不得把cap tokenrate称goodput。keeper419原metadata失败保留；CPU422独立审计、32owners和draft正常退出及post空已闭合，无GPU回放。
- **Kimi下一步：strict W8当前verifier的最短归因与一个局部候选。** 当前C1 verify inclusive占cycle约86–91%，尚不能归因为MoE/算力/CCL。先绑定当前DSO与已有trace入口，在下次有实际候选的正当load里采两PPstage各8个代表cycles；没有候选先做同源checkpoint小组件，不为报告重复约30.6分钟冷加载。重点验C16与C32前端并发；原生cap16的排队必须全计入，不能冒充32驻留。

本窗口没有新的可采用整Model性能提升。BW1000精确物理roof、各模型当前GPU/CCL/PP分解仍UNKNOWN；HOST CAPI并非GPU忙碌率，dense BF16峰值也不能替代FP4小T实测。

具体owner、最小改动、预热样本、质量、ETA、退出与替补见同名JSON。root需登记各卡采纳/暂缓/拒绝并派发；Deep独立准备不等本审核。输出token允许不同，最终以完整任务质量、真实延迟和每用户/全卡吞吐判定。


