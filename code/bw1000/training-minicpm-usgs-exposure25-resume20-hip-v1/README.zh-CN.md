# MiniCPM USGS88 独立25+5训练核心（尚不可派发）

本轮实际完成：对本机原88条训练JSONL（29658B/0a8560e6046439aa7450a3c1080ff5e537be17a56a8d4d332ff501d302b6bc62）及22张真实训练PNG逐完整SHA核验；构建800行完全保留原行字节的视图，88独立样本各曝光9或10次。未读取任何开发/盲测输入、金标或媒体，未改原分割或宣称新增800个独立样本。数据为原已冻结USGS ComCat公开数值的项目本地派生表格，沿原credit；没有新下载或官方原图声明。

`prepare_local_once.py` 已实际通过冻结T05共同 `build_launch` 构建 continuous25 和 checkpoint20→25 两个命令计划。MiniCPM 原 native template/model_type=`minicpmv4_6`、SDPA、关闭GC、冻结vision和aligner、4096全输入、8卡BF16/batch1/accum4、LoRA16/32/.05、AdamW/cosine/学习率预热5步保持。不是Qwen模型合同，也不修改正式T05原启动SHA。首次本机计划测试发现 resumed builder 已移除原baseline回调，修正错误的count1断言为0后两臂通过；未启动子进程。

`minicpm_usgs_witness.py` 从已封25+5共同观察器作窄适配：模型核验明确使用冻结 `vlm.runtime.minicpm_recipe_contract.validate_native_recipe(model,args,10822656)`（共同native_recipe_contract不支持T05）。实际运行时必须从原初始43,466,515B文件恢复9ac0f645...命名参数，不能从旧已训练adapter开始；观察完整100次micro/rank、25次真实optimizer计数、至少一次真实参数变化及10/20/25保存。合法首步LR0允许参数不变，但optimizer计数仍须推进；全25步没有参数变化拒绝。恢复复用共同安全loader、参数/optimizer/scheduler/各rank RNG，观察原生skip80及后20micro，比较连续尾段，不把计划当真实恢复。

`mid_epoch_recovery.py`、更新门禁及恢复比较器沿已冻结共同实现；除独立witness目录/schema外不改数学。新增T05模板/native输入资格由先前独立CPU88包负责，本包 `require_native88` 缺真实独审pin时硬拒，已在本机确认。原151恢复漂移FAIL和旧UNKNOWN保留。

当前只是经过本机真实输入与命令接线核验的训练核心。没有目标安装、原生encode、模型/权重读取、Torch/GPU或optimizer调用。原CPU88未实际执行；完整root端有限Slurm/held-child/RSS/强身份和清理driver尚未接入，建议2400/2100/1980只是预留预算，不能称已执行限时。因此全部GO=false、READY增量0。root无需发布尚未闭合的核心依赖。

下一动作：root恢复既有认证后先按原认领唯一执行CPU88及独审；核其完整输入/ChatML后缀受监督EOS/provider后填真实pin。随后将此核心接入已合格的有限控制器，普通发布新固定request b88f477d9e6f4e7d9e251c7090463b4f、fresh资源准入，才可实际25+5。连续训练或恢复任何失败均保完整分母；效果、正式数值和性能验收另外进行，不由loss或CPU计划宣称通过。
