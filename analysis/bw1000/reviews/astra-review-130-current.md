# Astra 130 紧凑审核

新增可证明整模型/真实服务速度收益 **0**，三模型硬件峰值、算子和系统优化目标仍 active。32K、普通4096/thinking8192、HTTP≥8K/SSE≥16K、实际8工具步、长参数 actual 与 ordinary 各≥1024 tokens、C2隔离和正常退出标准全部不变。

129已促成Deep1201构建闭合、1203一次加载和真实8K/16K连续输入EOS；这是真实执行进展，但三个regular工作流失败，不能算正常agent质量通过。GLM复用795转正式801；Kimi复用462/465和466/467进入471。G129F2的重复GPU0初始化修复未实施，r191已明确延至首次正式结果，应将implementation129登记为PARTIAL并引用此延期依据。不要打断801。

本轮可避免失败的共同问题是静态/独立CPU证明未覆盖实际入口组合。GLM798是同进程private protocol被旧导入污染。Kimi471已定位target_model.py:26向StageControl传8500，而同SHA helper只允许1..7200，通信、Native构造、权重前即失败。原owner已获r194有界修复授权：单collective7200，整体8500及验收1800/3600不变；复用现有CPU合同核对实际helper，不增加框架或重复GPU探针。

Deep报错来自工具解析验证层。public prepare_agent_request已按named choice过滤到指定工具schema，不能说named choice被完全忽略；是否仍缺必须调用/单call约束须由原输出判因。worker已保存generated_token_ids，实际bridge排EOS1后整段decode一次，HTTP strip后feed(final=True)，SSE feed(text)再feed('',final=True)。先恢复这两条真实路径；逐token分片仅是离线诊断，不能称1203实际分片。不能删除严格验证、丢第二call或伪造工具结果。

下一步按首判耗时安排：

1. Deep：一次既有token/wire离线因果定位（约5–15分钟）；仅证据支持后修编码或parser。保留1203的自动regular→long-output→销毁，不假设可常驻追加，不为微改重载。
2. Kimi：r194最小启动常量修复与既有CPU合同；成功后一次必要正式load，实际8K/16K成本直接并入完整验收，不重做Native/cache probe。
3. GLM：继续消费801同一句柄，完整验收或首异常决定后续。重复preengine/GPU0初始化下次必要启动再处理，覆盖新preweight和旧preengine两条路径。

完整五张实验卡、必要检查、资源、退出条件、替补和时间口径见同名JSON。没有新性能配对；累计运行时间与CPU步骤秒数不可冒充加载、测量或端到端成本，缺失项写UNKNOWN。当前审核仅是建议和证据分层，不是实施完成或新的GPU授权。
