# InternVL modelroot17 单 compute 组装 CPU 源码

状态固定 SOURCE_PREPARED_BLOCKED_BY_FAILED_PREREQUISITE，execution_READY=false、全GO=false。原3stream当前已失败，receiver03=1087656.366 CANCELLED0:9，04=1103606.194 FAILED1:0，源进程清理边界另列。不能执行该组装、不能为READY构造必然失败调用。将来新合格权重源必须显式修订新source/claim identity；不接受CLI换request/pub/epoch/nonce/路径。旧scope不改，0body/fixture/remote/Git，未重测任何已过资格。

## 精确依赖

DEPENDENCIES.json冻结361原14 manifest/原3stream PLAN/MF/361独立资格结果的exact SHA，以及root提供的原intent434B/dbc1780...精确身份：pub0c962c1deb86497f5efbcb80230037783512a3bb，request328b0f23c38949f1a9149f0006e48b9b，epoch1795057383.2257993/deadline1795143783.2257993，MF a36dbe5c23f495713b118793b92fde3cc7014d424ce7339574896e28cdc6318f，nonce549a9cbfa6e1473ab85a2d8bf712666c。作者读取原intent返回PermissionError后未重试/升级权限，原身份由root完整提供，不猜字段。

launch在任何新SSH、安装、Slurm和asset body前，必须从固定原dispatch目录读small稳定metadata，验证该同一原scope closure/producernormal/独立consumer全部3/3 PASS、11590blocks/12150388664B、6条正常SSH、3条独立fresh原sacct C0、各原source/receiver整SHA+wireSHA与FD/fsync证据。当前原FAIL/缺项/UNKNOWN/权限错误均立即failclosed，不能根据文件存在/相同pathname或新源PASS替代。该检查只是原强前提metadata承接，不重跑oldreader/collector/测试/传输。

成功前提才生成<=32KiB pinned upstream snapshot；其完整SHA进入新context，installer/host/worker/新independentconsumer都再次核同原身份与实际3PASS事实。current事实BLOCKED不会因本source完成变成READY。no-replay label/新namespace/PC attempt intent/fresh parent step去重；原缺项不触发fallback或自动重试。

## 唯一有限动作及字节分母

固定 parent1087656/nodea14r3n04，一次alloc4CPU/worker2排0,1、4GiB/0GRES、nice>=10、4GiB AS（不是RSS）。单compute observer先对原3stream的3个固定最终shard逐件read once wholeSHA、regularowner/EOF、同observer前后完整FD/path stamp；3件全过之后才允许读旧14或写任何新14payload。三个原件保持只读FD到scope末尾，再核不变。不能据跨HOST st_dev或pathname认为权重相同，这里用该compute新完整SHA绑定实际bytes/host/boot。

随后仅361原hash_names列出的14件，读取路径为 old_namespace/name（不是payload/name），单遍sourcecopy/hash，写3stream modelroot下14个未存在的固定新文件。每件独占assembly-incomplete文件，filefsync/完整FD-path稳定、pre-rename dirfsync；保留FD跨rename承接dev/ino/size/mtime，再核postrename FD/path完整5字段稳定，postrename dirfsync，source/finalFD均留到最终核验后关闭。rename自身ctime可以变化；所有只读输入ctime始终严格不变。原14不chmod/覆盖/链接，三新shard不复制，失败留本scope新部分文件与完整失败分母，不清理旧partial。

| 成功时分母 | 字节 |
|---|---:|
| 3新shard只读/hash | 12,150,388,664 |
| 旧14单遍copy source read/hash | 4,922,323,586 |
| 总asset read | 17,072,712,250 |
| 14个新目标write | 4,922,323,586 |
| 最终模型文件 | 17件 / 17,072,712,250 |

worker一个预分配1MiB bytearray，用POSIX readv和memoryview同步hash/write复用，无双份大block、模型/Torch/SDK/解码/PC payload。输出hash在写流中计算，不二次读outputbody；其额外hash输入CPU成本与实际IO分列。总read/write计成功syscall逻辑bytes，物理IO与throughput UNKNOWN，不以900s预算编造速率/完成预测。progress每>=60s或>=64MiB，以及末次一次，不每文件/块fsync progress；必要file/directory fsync不省略。

只逐清单17路径，不rglob存储、不扫26/8initial/ref/旧partial，无CO3D/其他被拒绝payload、A100读取或权重传输。本地仅小metadata/source；安装使用短普通SSH命令+stdin EOF小源码包，没有后续二进制stdin，因此不会混解释器源码与payload。

## 完整单epoch闭合

新900s T0从launch main首行起，prepare60/admission90/body760/stepend790/worker-hard-tail800/host815/held835/accounting855/client865/freshread882/fresh-held886/consumer890/PC895/whole900。Slurm用divmod HH:MM:SS、向下取整分钟并预留15s，worker body=min(T0+760,plannedstepend-20)。原epoch不足STOP，不续时、不增加step；原parentfreshUID/RUNNING/lease900/node无冲突、exactlabel去重后才一个srun。提前终态即退出，不等待900s，不建service/timer/poller。

new host持有srun新PGID，用qualified exact held_child清理owned对象；tail硬alarm覆盖FDclose和小结果写出。producer与原metadata结果分别保留。新进程read_result.py已在launch正常/失败路径实际接线；只读本次小receipts、snapshot与singleobserver17证明，再对已知新assembly exactstep+label仅一次普通fresh SSH/sacct核C0/cpu4mem4GnoGRES和held正常reap。原3stream不作新的freshquery/collector/reconcile。独立reader不读asset body/旧资格、不把failed producer升级PASS；原前提失败时无新SSH/Slurm/freshsacct。

源/目标同observer严格5stamp，meta写/fsync及read/hash/write成本单列，不以分配4CPU假4busy/以CPU占用冒GPU。正式17case/native/model/training/quality/performance GO始终false；17文件身份组装通过也不恢复原3stream失败事实、不构成全部26资格。

入口launch_once.py --publication <本包40hex> --source-manifest-sha256 <本包64hex>，当前禁止实际invoke。verify_package.py只做新source AST/MF/静态预算和元数据算术，不执行旧资格、dependencygate或worker/fixture。root独写status；本包task同样BLOCKED/0调用。
