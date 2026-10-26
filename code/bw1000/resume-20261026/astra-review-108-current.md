# Astra108

12个指定源SHA均通过。消费719、B735/495/574、Deep1096：2轮Model、1轮组件。A与724继续，未消费。

- Deep flat readback负结果PARK。立即D108A：复用1096，readbackOFF，匹配deployment tile16，cold/warm CPU调用分解并与无观察器wholeCSA配对。310ms是43层metadata总和，9.6ms是单CSA，不能相乘外推Model。先诊断，不盲写bank/std::set。
- KimiA独立32个C1目前30/32>15、全正确EOS，仍在C8；33卡低并发不能外推16卡或高并发。慢1218删尽其他成本仍13.07tok/s。保持其他成本时verify须14.234→<10.270s，或useful/cycle3.875→>5.114。主矩阵后且≥600s余量，仅做现resident W2/W4对W8；随后CPU做接受量/完整cycle成本策略。
- KimiB C16 state/AR +66.11%、C32 +26.51%，不可冒称adaptive/fixed收益；没有用户>15。
- GLM719 decode +17.36%，用户仍不到15。独立prefill724继续原配对，不新增无实际profile支持的算子路线。

详细owner、资源、样本、ETA、退出和替补见JSON。CPU文件轮询暂缺只修有界stat/read重试，不重启GPU或算性能轮次。无新framework、旧负路线重跑、F64或token逐位门禁。
