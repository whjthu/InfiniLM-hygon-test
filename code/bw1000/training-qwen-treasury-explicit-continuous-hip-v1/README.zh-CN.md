# 显式输出指令的独立 Treasury 五步训练

本包继承原 v3 的八卡训练控制器、共同框架 a68952cb、原始底座和初始 LoRA。唯一训练变量为 53 条 signed_difference 用户消息增加已冻结的格式指令。160 行、72 张图、答案、图片路径、顺序不变。batch1×8卡×累积4、BF16、SDPA、梯度检查点、4096上限、LoRA16/32/0.05、AdamW、cosine、seed/data_seed42、原学习率与 warmup 全部保留。

只有原实际 CPU326 合格后才允许训练。prepare 同时核验目标 CPU-v2 原 outcome、intent、step.log、源清单全部成员及实际 sacct326，与原 stdout 收据相同；公开摘要不能代替原件。新输入独立安装到新 RUN/inputs.jsonl，旧图的 /refactor 路径和挂载不变。

根普通发布后，将本清单全部文件安装到 R/results/training-qwen-treasury-explicit-sft-v1/src，新 JSONL 只安装到该 RUN/inputs.jsonl。根以绝对 HOST Python 调用 launch_once.py --publication COMMIT --claim-sha256 SHA --request-id 新32hex。全链只有一次1620秒epoch，独占intent和新label；原285/286须实际终态，随后 fresh父作业/同节点步骤/PCI/KFD/port/ownedPID 核验。8DCU工作后独立0DCU post核验，原bounded accounting helper保留。UNKNOWN保留原目录和句柄，不重提。

真实5次更新后停止，记录2/4/5检查点及全部8rank输入、loss和更新状态。初始化回调恢复原初始adapter，不能续训旧checkpoint。旧5step臂只有实际相同起点、顺序、消费与预算证据匹配后可复用；否则须另scope跑匹配旧输入control。初次本地配对命令比较只漏替换env/required_assets中的dataset路径，已保留记录并修正归一化；实际配方没有变化。

后续base/control和开发评分由根另冻结scope。保留所有wrong/missing/invalid/cap；signed_difference无改善或其他主题退化则不晋级，不做cap/prompt网格，不因loss下降加步。旧恢复失败分母不变；本包没有恢复或正式T01 SFT、三轮性能、盲测效果资格。原v3 collector曾读挂载延迟导致false，本包保留其逻辑；需独审真实GPU/八rank/post原件，不能把collector失败清洗为成功。

发布前已修正 controller 仍接受旧 RUN 的门禁，实际抽取初始化语句验证新 RUN 通过、旧 RUN/外部名字/外部父目录拒绝。全 Python 源没有旧训练 RUN 字符串。原 1620/900/1080 秒上限不变，每个新 step 的客户端等待和真实 Slurm 时间均受同一原 intent epoch 剩余预算限制；pre/GPU 保留150秒，post保留5秒。记账查询最多14秒且不超过原epoch，wait_terminal最多40秒且预留5秒写证据。原epoch耗尽不新建step，UNKNOWN保留原请求与PID。数学和训练plan未改，原v3保持原字节。
