# O01 八卡见证器的 HOST 派发合同

本包保留已有 GPU request e23d24ea0c9c4d00af06154d82dc22e8 和原 O01-zimage-lt-eight-card-supervisor-v1 标签。通过了 12 CPU 的零卡 host .2105 和 Slurm 进程归属 .2108 两件封存原始摘要。此前 .2075 仍处于实际运行。只有旧步自然得到 f12r2n11 的唯一 COMPLETED 0:0，且现场看到同节点没有其他活动编号步，才可准备新独立 pre。代码不执行 srun、scancel 或 SSH。

## HOST 接口

根协调者在现有 .training80-dcu-standby.lock 中调用 prepare_pre_locked。现采完整目标相关 Slurm 记录。该函数先固化独占的 pre intent，再返回单节点 0 GRES、2 CPU、4096 MiB、60 秒命令。调用方按原 node_inventory.py 在目标记录 boot、八 PCI、KFD 目录和项目 UID 的设备 FD。pre 与后续 post 使用不同 request、标签和路径。任一命令送达状态不清时在原 intent 和原编号下对账。

prepare_gpu_locked 在同一共享锁协议内重新采集。它绑定已保存的 pre intent 原字节、原前任终态，当前无在途编号的节点、独立 pre 的终态、同 boot 的八 PCI，以及 KFD 和项目设备 FD 都为零的原 stdout。运行时、固定 attested 入口、原 node_inventory、.1982 DSO、.2105 和 .2108 两个原摘要逐件核 SHA。验收完成后才设置一个 240 秒 epoch，独占写 scope、完整准入记录和原 GPU intent，再返回唯一 srun 构造。命令参数是 1 节点、1 任务、12 CPU、8 DCU、线程绑定和暂定 24576 MiB。24576 MiB 来自原 16 GiB 保底加原 8 GiB 探针的资源预算，目标实际可申请内存及峰值仍待核实。

根执行器须持共享锁穿过调用、唯一提交和实际编号可见时的归属绑定。不得用程序调用返回、目录缺失、空队列时间或 SSH 超时代替真实调度状态。现有 fleet 启动器只根据可见 Slurm 活动步筛节点。它需要由根层同时保护本节点的持久未决 intent，防止 HOST 退出释放锁后抢在尚未入账的提交之前补位。根执行器仍需接入这一保护和截断到原 epoch 的外层有界等待。设备入口从该 epoch 扣减，剩余不足 185 秒即拒绝启动 GPU；对账可以随后只读进行。

本包的 device_host.py 在实际 Slurm task 原 PID 内调用发布的 attested node_entry.py，先严格要求 kernel cgroup 命名中出现相邻 job 和 step，再进入原 attested host 路径。实际有卡步骤由原入口继续核对完整步骤、8 DCU、12 CPU、listpids、boot 和指纹，容器内核对同一归属。adopt_container_owner 按两份私有收据原字节对账。零卡成功本身尚未验证 GPU cgroup 与设备持有者。

## 清场和保底恢复

收到同一编号的真实 GPU 终态后，prepare_post_locked 核对原 scope/intent 并独占写后置 intent，返回执行一次的独立零卡命令。可以记录已终结的 GPU 失败，设备性能和模型验收仍无结果。refill_eligibility_locked 绑定 post intent、实际独立终态、同 boot 八 PCI 以及 KFD 和项目设备 FD 清零。根随后在共享锁内独立准入一个新普通八卡保底范围，使用现有 standby worker。consume_refill 要求新的精确编号和实际 8 DCU/8 CPU，两个观察点的 PID 与 startticks 相同，且八张卡的实际 BF16 GEMM 计数均增长。节点侧脉冲按同节点时钟校验，HOST 收据新鲜度单独检查。整个 GPU 步内部八卡恢复和下一普通保底任务分别记录，换步空窗仍需实际测量。

## 本机入口与准备字段

    python3 -I -S -B host_scope.py --evidence EVIDENCE.json --evidence-sha256 SHA --launcher-sha256 SHA --uid UID

该入口只读取证据，打印范围和 argv 预览，不创建 intent。生产函数从同一模块导入。scheduler 包含同一 HOST 时钟的 epoch、实际 parent 字段、已展开 parent_nodes、完整七列 accounting_rows 及 accounting_columns，以及已展开范围的所有 numbered_steps。accounting_complete 和 numbered_steps_complete 由实际读取器按成功完整命令记录，不能用缓存补齐。前置或后置 inventory 传入 request、原 stdout 及其 SHA、已发布 source_SHA256、同一 HOST 时钟的 start_epoch 和 observed_epoch。生产入口还核各原意图 SHA。

固定 HOST stage 内放 device_host.py、原 node_entry.py、原 node_inventory.py、原摘要副本 host2105.json 和 owner2108.json，以及已生成的 runtime/。部署、派发、实时归属和回填证据仍由根处理。此源包不晋级 GPU 准入。
