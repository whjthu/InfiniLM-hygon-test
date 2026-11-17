# Treasury 时间序列原生 CPU 输入资格准备

这是 Treasury 同一路线的独立输入资格包，领域候选仍只有 Treasury 与 VGGT 两条。原17正式用例、原160训练和108开发输入均不改变。本包不训练、不生成、不评分，不增加第三个模型候选。

复用已发布的数据准备包 `training-treasury-timeseries-data-v1`，不重生成数据、不下载。美国财政部 FiscalData Debt to the Penny 为原始数值发布者；原说明中的官方 Licensing 允许复制、改编、再分发及商业和非商业使用。48张图是官方数值的本地 Pillow 派生插图，不能称为财政部原图。许可说明与原查询、单位、字体及版本身份继续以原 SOURCE 和 DATA_QUALIFICATION 为准。

固定120日期、24个五日期窗口、48媒体、432题；train/dev/blind各40日期、8窗口、16媒体、144题。三个任务每split各48题，同窗3字段×3题型×2媒体严格共组。train为2025Q1首40日期，dev为Q2首40日期，blind为Q4首40日期。日期/窗口/媒体跨split无交集的原CPU数据资格不替代本包目标原生资格。新检查再次核48媒体bytes SHA、24窗口及432ID隔离；不读取dev/blind labels或windows内容。训练labels真实bytes SHA及assistant答案一致性会核验；dev/blind labels身份只引用冻结manifest，实际bytes SHA未核、内容未解析，报告逐项明列。

`check_native.py` 使用与统一训练框架既有 Treasury CPU 入口相同的 `swift.model.get_model_processor(model_type=qwen3_vl, load_model=False, download_model=False)` 及原 `qwen3_vl` template。固定Swift4.4.2、Transformers5.12.1、qwen-vl-utils0.0.14和模板源码SHA，max_length4096/max_pixels262144，truncation_strategy=raise。数据接口只投影messages/images，id/window/topic留为私有证据，不传给训练器。

目标一次检查共432次完整样本encode，另144次训练user-only前缀encode，共576次；不截断超长样本。训练144条核原生user-only前缀和图像token均-100、实际assistant监督、监督值与input_ids相同；开发/盲测各144条用generation模式，必须无labels，绝不加载其金标来构造监督。按split保留有效监督token总数和长度范围，失败保全部计划分母和完成计数。随后原生meta RoPE dummy model与两条不同长度训练样本完成collator，核右padding、attention、完整labels与图像grid，拒绝实体模型参数。

记录所有48原图尺寸、原生image_grid_thw、processed空间尺寸及pixel张量形状和有限性。它们只是缩放/接口/可读性代理证据，语义可读性明确UNKNOWN；不以图像token存在断言长金额在缩图后可辨认。实际模型能否正确读取金额、训练数值、连续更新/恢复、同预算性能、开发及冻结盲测效果仍待另行资格。

本地仅执行纯stdlib数据绑定和3正常stub/16失败边界。没有Torch/Swift原生调用或新CPU资格运行。本包只提供可执行目标checker和精确资源声明，不提供远程安装或提交入口。根协调者发布、核真实runtime/providers/模型及处理器资产、无GPU分配/可见性、当前namespace去重后，才能把checker接入既有bounded CPU执行器。原整个epoch600秒，Slurm不超过480秒、inside不超过450秒；实际srun分钟向下取整并随同一epoch缩水，client/terminal/post亦留预算；耗尽不启动，不延长期限，UNKNOWN不重播。checker内每条输入核同deadline，阻塞调用还须由原bounded父进程450秒上界终止，不以回调代替外部超时。

后续目标命令为 `python -B check_native.py --root <只读数据根> --model-dir <原冻结Qwen本地模型/processor根> --output <新的私有独占结果目录> --deadline <唯一CPU epoch原deadline>`。需先实际供给本次432题与48图数据根，不能沿旧CPU160/108 receipt宣称本次已通过。dev/blind标签不需要部署到执行器；manifest可原样保留其身份哈希。

可训练状态为未就绪：本包仅准备目标输入CPU资格；实际native encode/collation、语义可读性、训练/恢复、领域效果、正式A100同口径性能均未验收。即使目标CPU PASS，也不能晋级为模型或领域效果PASS。保持先开发定标、后冻结盲评；不因新数据改变现有3臂评测或拿盲标调参。
