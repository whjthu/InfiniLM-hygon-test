新独立train-only CPU前提源码，当前未采用、未发布、未执行目标。原 `.511` 是有图72行资格；本包的无图转换/encode/collator资格不能继承。所有root/CPU/GPU/training/Model/formal17/READY均false；本机仅mock验证，没有读取真实训练行、dev/blind、私有媒体或此前被拒绝的本机路径，也没有远端/Git/模型调用。

唯一未来输入固定为 `R/results/training-gemma-usgs-render72-native-CPU-v1/data/train-swift.jsonl`，54246B/SHA a897e7d9e7823d01bd2b503104d138f4a50975cabc06a2fefcda1dfb36ff05fd。来自实机 `.511` 已合格的train-only副本，不访问原数据目录、不用别名绕过旧审批。worker没有任意数据路径参数，不读data-binding、dev/blind文件或任何媒体body。精确源SHA锁定原72 IDs/12图引用；未来实际转换才生成ID顺序hash及逐行/问题/原FINAL标签/图引用hash witness，当前不虚填原ID清单或派生SHA。

公开renderer `training-USGS-multievent-render144-v2/build.py:51` 的输入固定为 `<image>\n`+question。transform.py只删这一个严格开头标记和images字段，其余问题字节、assistant FINAL、ID、顺序、kind、group、event metadata原样保留，不strip/rephrase、不造blank。重复placeholder/其他媒体marker/图引用越出train/缺ID或覆盖不完整直接失败。来源必须完整72/12chart/3closure组/6kind每种12；12图只核引用，不重新读取图片。它是同一训练集的输入信息消融，不是新独立事件数据；无图问题可能不可从问题本身回答，后续控制只用于区分格式/任务先验与视觉信息，不冒真实通用QA数据。

最小未来CPU入口（需另行Root审查采用/普通发布/新scope准入，**现在不运行**）：
`python worker.py --epoch ORIGINAL --deadline ORIGINAL+390 --publication EXACT_COMMIT --source-manifest-sha EXACT_MF_SHA`。
固定新request `eef6ad53dab54c3994d70b3adf25a4fd`，parent1103606/node f13r1n12，4CPU/8GiB/0GRES，whole480/Slurm<=420/native<=390，原epoch含准备与执行不重置。输出固定新 `R/results/training-gemma-usgs72-question-only-native-CPU-v1/audit` 必须不存在，已有意图/未知/终态不能重复invoke。worker限制本进程实际affinity4、Torch线程2；8GiB由外层Slurm请求/独审保证，本包不编造cgroup/RSS硬峰。

未来Root应沿原 `.511` 绝对Apptainer/DTK容器及已资格held/accounting/cleanup前端接此worker，不能原标签重跑或直接调用旧硬编码64版本。仅只读bind：这个新RUN/src、上述单个target train文件（同绝对路径）；原 `R/results/training-gemma-processor-assets-v1/models/gemma-3-4b-it`→`/gemma-processor`（恰好14 metadata）；原 `R/results/training-qwen-a68952c-swift-initial-CPU-v1/framework`→`/input-framework`（393精确manifest）；原Swift runtime→`/swift-runtime`、原Transformers site→`/minicpm-transformers`、原system alias只读；新RUN可写并保持绝对R路径。**不挂图片、完整模型、初态、开发/盲测或整个旧DATA目录**。环境仍source `/opt/dtk/env.sh`，三GPU masks=-1、HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=1、PYTHONPATH=/minicpm-transformers:/swift-runtime/site:/input-framework，CPU前端不用GPU --rocm参数。本包不包含HOST/SSH/Slurm launcher或自动派发器。

worker先核源MF/新CLAIM、target/node/原时限和隐藏设备，再在新audit下落原intent，核源train和14元数据/393源码、原实际版本。原生 `get_model_processor(gemma3_vision,load_model=False,download_model=False,device_map=cpu)` 必须返回model=None，metadata namespace不含weight；full max_length=None对照512/raise、max_pixels262144、right padding、train mode。72全部不截断、完整真实labels/mask、实际native suffix中EOS被监督，collator不改ids/labels/有效attention；encoded/collated像素字段即使None也拒绝。新路径如ABI不支持，保全72分母/实际attempt/failed/missing，不补图、不过滤、不换模板。

源与派生train JSONL、逐72 provenance和字段hash只写target私有audit；不会把sample tokens/labels或媒体写Git/回传stdout。进度在每次真实encode/collator前记attempt，成功/失败分开；native例外按行保留，超时/内存/阻止设备初始化保留partial与缺失分母。原.511的check_sample函数逐AST复用，其他新增无图条件必须本scope实证。记录实际输入/监督token，不宣称与有图等token。代码不显式查询is_available/device_count/SDK；Python lazy_init被阻止并计数，内部driver初始化仍UNKNOWN，不以mask或Python状态冒完整设备监控。

当且仅当未来实际CPU72全通过，才可讨论新训练arm的800view、e6目标初态复用、25真实更新及独立效应验证。本包当前无800训练视图、训练更新或GPU资格，旧正式T02/A100原初值审批完全不变。来源许可沿已批准USGS官方公开数值与本地派生图：`https://earthquake.usgs.gov/fdsnws/event/1/`、`https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits`；不增加下载，不称USGS官方图/正式benchmark。
