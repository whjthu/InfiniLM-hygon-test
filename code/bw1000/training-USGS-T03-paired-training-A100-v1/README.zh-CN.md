# InternVL 原配方的领域与通用对照训练

同一份404文件的框架保留全部17用例入口及T04实现，新增T03显式opt-in。原396框架、已运行T04的400框架与本次5文件覆盖由三个清单精确关联；不另写训练循环，仍由原Swift/Trainer执行。

先唯一执行domain25；只有实际C0、8个rank完整终态、独立核验和同8卡清理通过，才允许唯一generic25。每臂8张A100-SXM4-40GB、BF16、每卡batch1、累积4、有效batch32、原SDPA/语言侧梯度检查点、LoRA r16/a32/dropout .05、504张量43,646,976参数、原AdamW/cosine/warmup。新领域负载不是原COCO性能验收，不能据此宣称海光吞吐、正式恢复或任务效果达标。

复用各45条已完成原生预处理的训练输入；domain安装中一次生成两臂各800条的确定曝光视图（每源17或18次），generic消费同一已封存视图，不再编码45条或处理开发集/盲测题。真实每rank100微批、25更新及全局800曝光必须由原生运行记录核实，等样本数不等于等token数。原首步warmup的学习率为0，首次参数变化应在第2更新，不能把未变误作无反向。

各rank从原initial.pt的独立文件SHA恢复共同具名参数值及该rank的Python/NumPy/CPU/CUDA RNG，不能以seed替代。每rank先后执行两次完整SHA扫描（共349,643,798B），以及两次weights-only反序列化（参数storage共349,175,808B，另含RNG和元数据）。总逻辑文件读取字节和物理磁盘I/O未测，不能把两次扫描字节当成总读取成本。全部8rank保存实际输入/token/梯度/更新/具名优化器moment与step摘要；保存10/20/25检查点，rank0在25步对真实保存状态作类型和值对照。原T03恢复漂移原因仍UNKNOWN，不能由fresh25洗去。

每臂单epoch900秒，从本地准备/上传之前设置，贯穿复制、输入视图、准入、加载、训练、写盘和清理：native820、supervisor停止850、held/tree清理880、alarm890、客户端895；原epoch不得重设。installer选择至少32至多64个可用CPU并排除0/1；原qualified_driver.py实际提供256GiB组RSS软采样，非硬限额/峰值。保留held Popen锚、PID/startticks/host/boot、全部失败分母，原nvidia-smi完整8UUID/PCI/空进程/可绑定端口的新鲜准入及独立post。未通过准入不终止任何其他进程；本次不重复暂停历史SGLang PID。

root先执行seal.py --adopt及verify_package.py，普通发布精确清单、认领和状态；其后build_body.make_body只生成一次arm的stdin，root持久化唯一request/epoch/attempt后使用既有A100-27连接。原目录存在即停；SSH超时、未返回或强制关闭保留UNKNOWN，禁止同名或改名重放。generic所需独立domain终态不是controller自己声明。当前源准备不是实际训练成功，Model/formal17/效果/正式性能资格均false。
