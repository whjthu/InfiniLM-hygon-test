# GLM T512 Fresh KV 的两头 CTA 对照

原 `.1026` 实机验证显示，T512、TP16 每 rank 四个本地头的 BF16 pack，128 pair tile 约 9.0 µs/次，64 pair tile 约 15.8 µs/次。首段 T512 的真实 resident 调用已有 `.1157` 与 `.1180` 的完整调用证据。`.1180` 的独立模型配对通过数值与首段速度核查，keeper 终态仍为失败。本候选仅看该 pack 操作。

同一 TU 和同一 CUDA provider 下，H1 与 H2 都用 128 线程。H1 每 CTA 搬一个本地头，H2 搬同 token 相邻两个本地头并共用尾部 64 路 RoPE。T512 时 2048 CTA 降到 1024，名义 RoPE 读少 131,072 B/调用。缓存复用和指令生成待实机核证。两臂使用原始 `[512,4,448]` BF16 投影与 `[512,1,64]` rotary 形状，以及 `[512,4,256]` 两个输出。数学、地址互斥和写出字节一致；benchmark 使用有限受控数据。和旧 tile compare 的 H1 性能数字仅作原热点依据。

## 首次 build

使用本目录 `prepare.py NEW_SOURCE_FOLDER` 生成独占的四文件 staging，预期 manifest SHA `e588d6b8b2d0ba0e14c6dd89921ac1258f293f48fe6da90bc1b99b579652a59c`。根沿用已资格的 DTK26 CUDA 编译容器与 0-GRES 上游启动链，运行 staging 的 `compile_once.py`。传入已核准的 `TUNING_CPU_MANIFEST_SHA`、`TUNING_CPU_PARENT=1103606`、`TUNING_CPU_NODE`、`TUNING_CPU_EPOCH` 与 `TUNING_CPU_DEADLINE=epoch+300`。0-GRES 三设备可见掩码都为 `-1`，ROCM_PATH 仍为 `/opt/dtk`，4CPU/8GiB。原 .855 的 compiler prefix、1113 producer 依赖图、POSIX工具路径及 CUDA 头保 hash。fresh M 与 actual MD 闭合，目标对象后用同既有 CUDA runtime 链接单独比较 DSO，验证实际动态 provider edge 和两个 C ABI 导出。

输出部署至 `/refactor/candidates/GLM474-fresh-kv-headshare2-gfx936-CPU-v1`。根独审实际 0-GRES 编译终态和 DSO/receipt SHA 后才编制 GPU scope。没有该实物时入口拒绝加载。

## 首次小试及短分片

在审核目标源码旁按原字节安装 `routed-i128-one-gpu-caller-v1/hip_runtime.py` 与 `fresh-kv-pack-event-paired-v1/timing.py` 的已锁成员。根使用当前 surviving 1103606 父作业的 fresh device/ownership 证据，排除 I03 三个 Deep 节点并保留其他更高优先级实际预约。小试只允许 O01 f12r2n13 的八卡。每节点一个十二 CPU 的任务，八卡由实际 ordinal/PCI 绑定，4×5 秒 H1/H2/H2/H1。每臂保同一流计时、墙钟、每个已完成 CABI 与有用行计数及四个局部 driver 读数。在计时外核两套输入、回原输入、原 BF16 K/V 全字节、guard 和非法 C ABI 拒绝，保 8GiB。GPU step 受同一 epoch +150 秒约束，Slurm 三分钟，`SIGUSR1` 协作收尾只用于精确已拥有编号步。

由根先核并保留当前父、全局共享队列原请求、设备/端口及 fresh placement，再运行 `launch_plan.py ROOT_ADMITTED_PLAN.json`。它只渲染有限 argv。顶层提供 kind（pilot 或 followup）、request、package、campaign（以请求后缀结束）、epoch/deadline_epoch、project_owned、fresh_protected_excluded、build_receipt/其 SHA，以及按节点索引的 nodes。每节点记录 node、parent、request、epoch/deadline_epoch、pre_step、root_admitted 和 scope_SHA256；根在 `<campaign>/<node>.json` 安装与此对应的 device scope，含 PCI 八项、boot_id、build_step 与 build_source_manifest_SHA256。渲染的 argv 携带该 scope 的精确 SHA，节点实际 worker `device.py` 只在选中的 Slurm step 消费。

已知 Slurm 终态供 `read_terminal.py --campaign CAMPAIGN --actual EXACT_ACCOUNTING_AND_SCOPES.json` 逐卡独审。输入含 label、原 accounting_rows（七列）、expanded_accounting_nodes、scopes；无账单保留 UNKNOWN，原 namespace 禁止重放。每次 release 还需根独立 0-GRES physical post。完整八卡 16 对通过且两个配对方向 H2/H1 中位均至少 +3%，才允许带实际终态和独立 post 作为 pilot 输入渲染 followup。每 shard 上限四台，后继合计至多 32 台/256 卡且按真实 fresh 队列选择。实际模型时延、训练、整池 busy 和装载率均未由此证明。

通过本机 C++ 17 检查算术映射和无效输入，并通过 Python 检查 staging、缺 artifact、分片及失败分母。当前源码没有目标 GPU 或编译实效。
