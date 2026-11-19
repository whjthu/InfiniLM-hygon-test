# InternVL rank00 initial + small reference：唯一900秒完整传输源码

仅准备源码，PLAN/CLAIM root/provision/READ GO全部false，GPU/Model/training始终false；本包未执行，无权重/旧receipt读取，无远端/Git/fixture操作。原transfer两端已知终态FAIL另行封存；本任务新namespace不是未知重播，也不涉及Gemma或其历史denied scopes。

唯一payload：原完整rank-00 initial.pt 174821899B（SHA2a0df4d5…）加固定PC原reference25357B（SHAa52d848f…），合174847256B（约166.747MiB，<192MiB）。reference来源明确为PC精确小原件，一次base64放源bootstrap控制体，不冒A100原文件；实际payload仍binary，不存在删除perchunkRPC/base64的提速。initial源全SHA+稳定FD，relay逐文件独立全SHA，compute接收全SHA+完整写/fsync/稳定新inode。接收端仅对25357B新reference再读并校验schema，单独计入reference_postread_bytes；不二次重读174MB大文件。8rank输入reference身份不代表8rank initial/RNG已齐：只提供rank00原完整字节，其余7rank完全不继承。

PC入口relay_once.py --publication EXACT --source-manifest-sha256 EXACT。root必须独审后将PLAN/CLAIM root/provision及CLAIM READ五个开关采用并普通发布，才可唯一执行。epoch在入口首行、参数后任务main第一行建立，含普通发布核验/小source安装/两端准入/完整传输/清理/Slurm accounting/独立consumer，deadline=epoch+900不重置。local-only/<label>-original-dispatch与A100/目标新namespace均拒绝已有目录；Slurm新label再次拒重。任何SSH timeout/forced kill/255只保UNKNOWN，不以杀本地SSH证明远端终结，不重播。

两条payload SSH都用短python -c BOOTSTRAP；Windows commandline<30000字符。大source bundle只在stdin里以8B length+精确长度script发送，BOOTSTRAP用os.read(0,remaining)不超读，exec后等待READY/GO，不使用python -等待EOF。GO是3B，所有remote控制和payload读取都用raw FD，不混sys.stdin.buffer预读；GO与payload同包也只消费3B控制。PC source.stdout的readline/read1始终使用同一BufferedReader；PC向target stdin的控制写完flush后才raw写payload，没有悬置buffer。

目标SSH只安装小源码、调度并在内存里透传；payload全部写a14r3n04 compute新namespace/payload下，login不写大文件。target_host复用已合格host_launch的parent1087656 owner/lease/node/活跃step检查、label去重、held srun清理和精确sacct terminal口径；唯一srun -c4 --mem4096M --gresnone --cpu-bindcores。step从13min按同epoch剩余向下缩为11/12/13min，不足11min拒绝。node_entry重新核已发布源码和真实Slurm parent/step/hostname/UID；保持alloc4/worker2，排除CPU0/1后仅选2个实际worker CPU，准备新目录及freebytes门槛后才发compute READY；quota未知，不能声称配额保证。源A100纯stdlib单进程/2CPU/RLIMIT_AS4GiB/nice>=10/0GPU，held READY前只stat原initial，不读取174MB；只有PC收到真实compute和source READY后才发GO并打开源initial读全体。源/目标均不import Torch/SDK或解码张量。

work截止不晚epoch+800，并受compute step_end-15缩短；host stop835、held清理855、accounting875、普通客户端885、local consumer与清理895，整体900。源4GiB是AS上限而非RSS承诺，target有Slurm4GiB及AS4GiB；源/target记录self maxRSS，不能冒cgroup/group peak。760秒payload需约0.220MiB/s只是算术预算，真实可用窗口以release时间和node deadline记录，无速度承诺。准备/分配/阻塞读/阻塞转发写/源盘读/receiver盘写/hash/fsync/progress receipt各口径分列，进度列whole/interval逻辑MiB/s，不把并行阶段相加冒网络独占，也不声称wire或物理IO已测。

实际独立consumer已经接入relay成功分支：producer outcome原件先独立保存，随后新本地Python进程运行read_result.py读取本次intent/READY/release/source-terminal/target-terminal/outcome小原件；对两件已知全SHA、全部字节、正常SSH返回及held清理作独立一致性审计；并实际另开一次普通Hygon只读SSH，fresh sacct -j EXACT_STEP，按原明确columns和exact step+label取唯一真实终态，与producer原accounting逐字段比较。该独审不提交Slurm、不轮询、不读payload/旧receipt/全proc，不以producer自报accounting替代fresh查询。结果另存independent-result.json、consumer stdout/stderr和closure.json；consumer未通过则relay最终返回失败，原producer结果保留。不是未来TODO；本轮仅静态接线和AST/MF，尚无真实接口/传输PASS。失败重消费须root明确以原已收原件独立只读审计，不自动再invoke。

原14资产、原partial和旧inode均零访问/零写/零硬链/零重hash；仅新namespace落两件完整新数据和小metadata。即使本scope成功，也只证明两件完整原字节可用，不证明all26、其余7rank RNG、原resume漂移、native provider、数值/训练或formal17合格。将来复用14仍需当前同观察者metadata身份门槛，本scope不替代该资格。

修订资源/独审边界：target始终Slurm alloc4/4GiB/0GRES，但worker2（沿已合格StageA接口），READY和consumer分别核alloc4/worker2。独审caller<=30s且收缩到epoch+895，fresh SSH/read<=min(开始+25,epoch+892)，自身held SSH kill/reap<=epoch+894；至少10s读取余量才开始，余量不足直接FAIL并保原producer，不重播。新fresh query有独立attempt/returned stdout/stderr/SHA及transport结果，timeout/forced kill/255保UNKNOWN。两条payload SSH加最多一条独审只读SSH共至多三次有限SSH；旧payload传输身份未READY或失败仍保UNKNOWN，预算不延长900。
