v2 窄修复（v1原件保留）：v1实际transport rc1，在controller读取预定intent前失败，根实核没有Slurm编号行。Windows str(Path('/public/...'))生成反斜杠，Linux创建了错误相对目录。本包只在新namespace重做此次未进入Slurm的启动前端，不重跑.845 staging，不重放v1。

新queue/transport/claim request：01f75d6e7c3a4d54928bfc8328a266ab；上游TRAINING_REQUEST仍247fa3682a114542ac3640fbc726aaae。local intent/context/receipt的request_id是前者，同时带training_request_id；target intent的request_id仍为后者，同时带dispatch_request_id。ROOT桥接按两种身份分别绑定，不把新attempt冒旧request执行结果。

launcher仅用ROOT.as_posix()序列化目标路径，并用PurePosixPath严格核绝对/public/home前缀和精确v2 label。target落盘前二次assert绝对/无反斜杠；先写完全部源和intent，再exec -I -S显式src入口。CORE沿真实已部署incoming/training-gemma-usgs72-fresh-initial-domain25-v1-source。容器Bash -c显式补positional $0=gemma-continuous25；原shell没有$@/参数依赖。squeue --steps使用已资格shared Swift同一%i|%N两个确切列，全非空行严格解析，拒额外/缺失列和不明step，不丢坏行。

本版仅新PORTABILITY_CHECKS.json三项实际Windows检查通过：实际make_body的POSIX常量；将同一receiver body在temp target Path shim落盘，截获最终execv时核intent已在正确目录，随后真实fresh -I-S解释器导入实际生成entry（拦截__main__，0Slurm）；非法根、Bash $0及steps坏列拒绝。不是Linux内核/ELF或目标执行资格。旧LOCAL_CHECKS.json/check_local.py只是继承v1历史证据，未重跑，也不据其宣称v2目标PASS。所有数学、helper、source/core/provider/权重/88等原件和预算无变化。当前GO全部false，根审/发布后才一次实际调用。

Gemma USGS72 新初值：一次连续25更新的运行封装

本包接在已经发布16081fc061703371d8e19012ded9e073a87f98c3、MF d3e8a643… 的原生框架包之后。原0-GRES .845实际COMPLETED0/22秒；stage-complete 151938B/f10d2775…，396 framework manifest40ed7488…，runtime2bf81432…。不重新 stage、下载、编码、读旧A100初态或改该已发布包。实际输入是既有USGS派生图 train72 QA/12图、冻结800重复曝光；不是800独立样本。dev/blind不读。新target-fresh seed42初值与原A100参考独立，正式17、模型/效果/性能均不晋级。

本包独立操作根：R/results/training-gemma-usgs72-continuous25-supervisor-v2。
沿原尚未执行的训练request：247fa3682a114542ac3640fbc726aaae。原0-GRES staging request不同，不重播 .845。
唯一原生输出保持R/results/training-gemma-usgs72-fresh-initial-domain25-v1/continuous/train。
目标1103606/f13r1n12、端口29932；必须根再次核当前排队/活跃owner及鲜PCI/KFD，不凭此前600算子清场历史准入。

根的执行顺序：

1. 审查本包并采用root/GPU/training/READ执行位，更新SOURCE_FILES.json后普通发布。原stage核心包/MF保持原样，不改权限或数据。当前本包所有GO=false，实际目标调用0，READY增加0。
2. 精确检查旧CORE路径R/incoming/training-gemma-usgs72-fresh-initial-domain25-v1-source及其MF仍为d3e8a643…；原ASSET根中stage-complete/396/runtime三pin和新continuous目录不存在。stage读取的旧model/data/framework须继续可见，不能碰旧A100 initial路径。
3. 通过原授权通道回调调用launch.launch(transport_start=ROOT_CALLBACK,publication=本包实际commit,local_output=新的私有独占目录)。没有默认SSH、helper、凭据或服务。回调只接收固定command、request、epoch/deadline、manifest/publication，返回真正subprocess.Popen（binary stdin/stdout/stderr PIPE，start_new_session=True），launcher喂原source-only body。固定HOST command是/usr/local/python3.12/bin/python3 -I -S -B -u -。source准备、上传、运行与清场共用函数第一行起的2400秒，失响应UNKNOWN不可重播。
4. receiver独占创建新操作RUN/src，交付的成员在-I-S下显式加入唯一src后运行controller。HOST keeper核parent RUNNING/唯一parent accounting、精确node membership、无活跃node owner和原三label不存在。全体非空accounting行必须七列；不得丢坏行冒“无历史”。pre为2CPU/4GiB/0GRES，原shared HOST snapshot核8PCI/KFD+同UID FD完整/端口，并保存原step/boot/单调时钟。
5. 同一origin只有一次8DCU/64CPU/224000MiB GPU步，Slurm最大2100秒，原生stop在origin+1980，worker清理边界+2080，whole+2400；actual Slurm分钟向下取整按余量缩水，不生成0/无限time。Apptainer绝对1.3.4、DTK env/已资格runtime与原Gemma data/model挂载，调用supervisor再exec已冻结CORE/worker.py continuous；CORE重新核fresh≤45秒PCI/KFD/HIP/端口及stage身份，真正走396 common Swift入口。
6. supervisor复用已实际资格的RSS-v2 held_child.py/owned_processes.py逐bytes、checked/read/sample RSS函数逐AST；只有精确新Gemma output+callback的native_command接线变化。GO前保存held PID/startticks/host/boot，registry按原owned树规则；VmRSS缺失记UNKNOWN且持续缺失拒绝，绝不补0。256GiB是采样soft stop，实际Slurm请求224000MiB不是该soft limit；采样峰只是下界且共享页可能双计，不是cgroup/内存硬峰。两路清理分别try/finally，held失败仍做owned tree，资源记录失败不遮蔽另一清理。stdout/stderr各16MiB上限。
7. native结束后核8rank各100真实forward/25optimizer更新/初值一致/10、20、25checkpoint记录。每个checkpoint原生必需14成员做稳定整SHA及cursor检查（只本scope新state、本机不导出tensor、不反序列化），记录cp manifest；不是说checkpoint目录只有14个文件，额外原生说明文件可存在。清理后必须GPU实际accounting终态，才做独立0-GRES post。post原8PCI/boot/KFD/端口相同，按保存同host/boot/PID/startticks检查owned退出。失败也post；原GPU终态UNKNOWN不释放、不取消陌生owner。
8. 原held transport RETURNED并不直接算独审。根在原f13r1n12计算挂载（0GRES、2CPU/4GiB、45秒）执行src/read_terminal.py --publication 本包commit --epoch 原intentepoch --source-manifest-sha 本包MF。不是登录HOST：compute和login st_dev不等，不能忽略该差异再宣称原checkpoint identity相同。reader无Torch/设备query/新encode，读原3阶段accounting、strong owner/post、8rank完整metadata、producer全SHA后原state五字段连续性；输出≤16KiB。原checkpoint tensor值只producer/未来native restore条件证明，不声称独立CPU typedstate反序列化。

真独审通过才在原ASSET根独占写continuous-independent-terminal.json（同fixedrequest、25更新/8rank/100micro、accounting_C0/owned_cleanup/independent_passed true和原checkpoint20的14 pins）。本包绝不启动resumed，也不会覆盖连续记录；后续恢复须另外让根审真实原终态/原epoch余量及新准入。实际初始参数/RNG、变化量和失败分母全部原件留目标，Model/formal17/效果仍false。

本机LOCAL_CHECKS：9项stdlib通过，无Torch、模型body、原生encode、Slurm/GPU或optimizer更新。覆盖原helper逐字节/函数AST、全七列坏行拒绝、时限、无默认transport、隔离-I-S交付成员可导入、held GO接线、RSS unknown不伪0、源挂载/输出匹配、native未启动失败consumer。首轮7/8通过的Windows execv PIPE差异保留私有证据；最终Windows fixture只截获末execv检查argv/token，目标Linux bootstrap未改，不冒目标实际执行。旧17项RSS资格没有重复跑。

USGS公开数值许可/派生源、分组切分和原.511 native72资格均来自原stage包；本包不重申新的许可或GPU/效果证明。来源与条款可沿原包README的USGS官方API和copyright/credits链接查证；模型仍受独立Gemma许可约束。实际操作前提是根当前正常认证/通道和计算节点准入；本包不绕通道或原拒绝的初态导出。
