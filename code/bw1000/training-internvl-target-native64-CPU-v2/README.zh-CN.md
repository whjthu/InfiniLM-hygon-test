# InternVL 目标平台必要 CPU 资格

这是尚未执行的 T03 新组合 CPU 资格，不是原 standalone version.py、sixshadow 查询或未知 fcafa 权重传输的重试。旧 v1 的任务记录为0调用；其中只有 agent 设置的 pending 布尔值，现有精确记录没有给出原用户禁令或自动审核拒绝 receipt。此事实不宣称其他历史操作已获许可或 UNKNOWN 已解决。使用用户持续授权的常规 CPU 范围，先冻结源码及认领、普通发布，再对新 namespace 唯一执行。

只读取原已传的13件模型元数据、完整64条原 HIP 输入及已知图像/插件、393文件共同框架和实际软件 provider。原 HF basename 固定为 InternVL3_5-8B-HF，仅复制13件元数据到新视图，不读取17GB权重、8份 initial 或旧接收器。实际运行原生 processor/collator/完整输入和 logits mask 语义，meta-only 构造原 InternVL/PEFT 结构，核504个可训练张量、43,646,976参数、语言侧梯度检查点和视觉冻结边界。Provider 由本次实际调用链测得，不能在本次资格之前又要求它先合格。

复用 a68952c 的393文件共享依赖，逐清单核当前字节；物理根为 R/results/training-qwen-a68952c-swift-initial-CPU-v1/framework。注册入口17用例的旧通过不代替 T03 runtime/provider 资格；A100 的404训练成果也不冒充目标平台已经部署。最终全部模型适配仍必须并入共同框架。

8 CPU、12GiB、0 GPU GRES；新单epoch300秒贯穿上传、安装、Slurm等待、容器、worker和清理/accounting，具体较短子预算以最终 PLAN/CLAIM 及实际有界控制器为准。原epoch不重设。隐藏三种GPU可见性；禁止显式设备初始化和权重/缓存文件打开，记录自身设备FD。Torch/库隐式设备探测若未独立测得，保持UNKNOWN；不能把CPU资格叫GPU数值或训练通过。RSS以实际组采样及Slurm内存边界分别报告，不假称采样为硬峰。

失败保留64分母、已完成行、阶段、错误SHA和原日志；held Popen、PID/startticks、进程树与Slurm终态独立核实。缺失、超时或SSH未知不重提、不改名运行相同范围，不终止他人进程。此包不查询六项历史 shadow、不清理未知receiver，不下载或重传模型，不执行训练前向/反向、图像嵌入、优化器更新或GPU算子。Model、GPU、训练数值及17用例正式验收保持false。
