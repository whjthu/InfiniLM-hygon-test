# 当前640卡单窗口观测候选

审阅时所有GO=false；Astra COMPLETE/PASS 后 root 已采纳 root_GO/CPU_GO，其他GO仍false。尚未连接远端、提交Slurm或采样；普通发布后才可唯一调用。它是新的当前窗口，不是重放原34e；原34e保持UNKNOWN，不取消、不释放、不读其namespace。

复用Nov16实测合格的 tools/sample_training80_busy.py 原字节，使用其node_map/collect/aggregate：3点、间隔2秒、4秒跨度、每点读取起止必须落在计划点后1秒内。旧collect_training80_busy_once.py仅作设计来源；146加载例外、旧owner读取和滚动期限均不继承。节点归属精确绑定当前active-node-task-map原字节，运行前还须fresh numericUID、两父RUNNING40/AllocTRES320、80节点唯一且逐父与账本一致、EndTime覆盖同epoch。

当前任何编号步骤或非batch/extern的未知步骤（包括TDB）都使此候选停止，不提交、不重试、不取消原步骤。去重同时检查新本地/远端namespace和两精确新label的accounting。不是用CPU采样制造GPU工作。

控制器只用2个允许CPU（排0/1），nice不提高优先级，AS1GiB。两个短CPU-only步骤各40节点，每节点1进程1CPU、256MiB、0GRES、60秒硬上限、--exact/--immediate=5且无overlap。合计最多预留4800 node-CPU秒、20GiB节点内存，另控制器最多360 CPU秒；这些是预算上限，不是实测CPU消耗或GPU活跃卡时。

每节点先用sysfs核当前8个DRM PCI与KFD gpu_id/render/domain/location双射，读原三点gpu_busy_percent后再次核映射不变。不初始化GPU、不importTorch/SDK/模型，不读权重或扫描KFD holder/PID。PCI可跨节点重复，卡键为(node,PCI)。本窗口的PCI是fresh运行时身份，不把旧PCI库存打新时间戳。

单epoch180秒在本机Git准入前确定，安装、父/步骤准入、srun、节点、记账与清理都同截止。开始新步骤必须仍有至少85秒（第二步80秒）；Slurm始终完整1分钟且不靠分钟上取整越界。控制器work在145秒停，保35秒给自己held srun的TERM/KILL/reap和有界记账；finally仍同原180秒硬退出。普通SSH强杀/255保持UNKNOWN且不重播。

完整观测应有80×8×3=1920个card-point；缺节点/丢点/冲突/超时/记账缺失均保留分母，primary current_driver_busy=None，不能以部分卡代全640。每卡的3点min/median/max/windowmean写R自有per-card-window-private.json；公共候选摘要给全640卡均值分布min/median/max、严格>80卡数、zero卡数、缺失卡数，并按实际账本各T/I/O任务和training/inference/operator分区列同口径分布。缺失卡不制造PCI标识。

busy是这一4秒窗口的驱动sysfs计数，不是SMoccupancy、训练进度、全天利用率或因果归因。共同wall点沿原合格测法，未额外资格化跨主机时钟同步；不得用跨host wall差称实测耗时。未实际执行前currentbusy仍UNKNOWN。

未来唯一入口（仅root发布并采纳后）：

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training80-current-busy-window-v1/run_once.py' --publication <exact40hex> --request-id <fresh32hex>
```

cwd必须G。没有新timer/service/poller，也不在每次heartbeat自动采集。原采样/模型/读取失败全部保留。新增纯CPU只验证本次准入与卡分布字段，未重跑旧采样fixtures。
