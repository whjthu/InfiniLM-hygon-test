# Qwen USGS：完整累积的 25 步训练和 checkpoint20 恢复

本任务复用真实完成 1103606.493 两次更新、494 独立清场的 USGS 88 条训练记录、22 张表格图和 Qwen 模型/原始 trainables。官方 USGS 数值及其派生图保留原来源署名；没有下载、重绘、新标签，也不读取开发或盲测输入、图像、答案。原493不重提，新请求固定为 `f572fda6784140cbbdcae7bb91a916b9`，独占新 namespace。

原88行在8卡上每 rank 每轮有11个 microbatch，不能把两步 witness 直接延长而宣称每次都是4次累积。新 data_view 从精确已资格源 SHA 复制完整原行，顺序循环至800行：前8个源行曝光10次、其余80行9次。独立样本仍88、图仍22；800仅是曝光数，不是新独立数据。统一框架的原生采样器仍照原seed运行；曝光 ledger 的物理行号不冒充 rank/forward 顺序，实际顺序由原生 forward 输入指纹记录。

同一8卡 BF16、batch1、累积4、有效batch32、4096、SDPA、语言GC、冻结视觉、LoRA16/32/.05、原AdamW/cosine。统一393框架按原manifest复制到独立目录；只复用现有 native T01 resume opt-in/cursor。初始参数从原绑定的文件真实校验、恢复，不以seed替代初参。正式17配方和原A100参考不变，本任务是单独domain recipe。

两个独立原生训练进程：continuous 25更新/每rank100micro，checkpoint10/20/25；resumed 从同scope checkpoint20完整原生恢复到25，再5更新/每rank20micro。总计30次全局更新调用计划，不能按8rank乘成240。每个rank实测其初始化、实际AdamW槽step、参数改变、full-input/loss指纹和实际监督token；重复曝光不假设等token。cosine末段允许某步参数值因精度不变，但原生AdamW每槽step必须实际递增，warmup 的合法零 LR 首步允许参数不变；全25步必须至少一次参数改变。

checkpoint20存储字段精确14项：adapter、optimizer、scheduler、trainer_state、args、adapter_config、8份rank RNG。原生Trainer载入后由ResumeWitness核参数/优化器/调度、空累积边界、各rank RNG，并实观察100-batch loader跳过80、余20。独立比较160个尾micro和5个尾更新及最终RNG；保持原exact诊断口径，失败保留，不重训。checkpoint10/20/25各14个文件的实际bytes/SHA和原生global_step另记录；本scope不把最后25的存储SHA冒成独立PT数值审计。

资源候选1103606/f13r1n10，8DCU、64CPU、224000MiB。whole2400秒含本地准备/Git/传输/框架与资产检查，GPU Slurm上限2100，两个native进程共享1980秒，不分别重置。各实际请求时间按原epoch剩余量收缩，保留accounting/post预算。记录真实step elapsed和各arm wall；原493的102秒不构成新30步上界。加载、指纹和checkpoint I/O全计实际工程成本，不作为正式性能轮。

`launch.main(publication, transport)` 使用root现有普通SSH回调 `(body,out,err,timeout)`，前台等待，不新建服务或隧道。唯一派发后UNKNOWN不重播。`read_terminal.py` 预置只读原metadata，禁止新query/processor/权重读取；原始checkpoint和输入都留目标端。若任一arm失败，25和5两个分母仍保留；连续25已完成与恢复失败应分别报告，清场不能洗成Model合格。

当前只有本地source/AST与新增游标边界资格。root/GPU/training GO均false；目标视图构建、native训练、断点、资源准入和效果均未执行。root审查、采纳、普通发布和唯一实际派发后，仍须独立终态消费；领域效果、正式性能、formal17和Model均不晋级。
