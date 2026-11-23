# Z-Image 探针与七张保底卡的前置检查

本包只验证新八卡 Slurm 步内的 HIP 视图。父进程读取实际八个 PCI，按原有 HIP/CUDA 选择器顺序派生单项选择器，原样保留 ROCr 和 GPU_DEVICE_ORDINAL。八个新进程各自加载已钉定的 HIP，必须只看到 ordinal 0，且实际 PCI 与父进程为它选择的设备一致。终态保留八个计划分母、子进程 PID 和退出结果。它证明的是目标 HIP 进程可见性。环境变量不构成设备访问控制。

根协调者取得一个新的、独立拥有的八卡步后，可以按以下命令执行。scope 需要 request、label、parent、node、fresh_root_admitted、boot_id_SHA256、allocated_DCU、PCI 八项、independent_pre_step、independent_pre_SHA256、epoch、deadline_epoch。调用方在 HOST 独立绑定真实八卡账单、当前设备清洁状态及单 epoch。实现拒绝使用 1103606.2075。本包没有 Slurm 派发或取消入口。目标 HIP 隔离仍待实测。

    python3 -I -S -B mask_preflight.py --scope SCOPE.json --scope-sha256 SHA256 --out NEW_DIRECTORY

验收阶段可以在新八卡步骤里先启动已合格的八卡保底 worker。监督进程先核准八张卡的两点调用计数都在增长，再用同一步骤的 CPU 余量另起上述前置入口。八个检查子进程只查询 HIP 数量和 PCI。现有八卡 worker 可继续计算到原有边界，检查失败也无需停它。两条路径同时运行时的驱动初始化和 CPU 影响仍需实测。此接法适用于可见性验收。保底 worker 当前以整个步骤协作停止，正式移交一张卡需改用下文的每卡子进程监督器。

## 同步骤共驻留资格

1. 通过设备前置后，将同一映射固化到新八卡步骤。实际业务子进程仍须在任何训练或保底 GEMM 前各自重验 HIP count、PCI、Torch count 及所有者。原探针和 hipprof 后代须继承同一窄掩码，核对自身 PCI 和 trace owner。八卡申请和卡号映射均经独立证据核实。
2. 七个子进程各在唯一 PCI 上复用已合格的 fleet-dcu-standby-v2/worker.py 的 run_device。沿用 8192 BF16 驻留矩阵、单流同步、每批 8 次调用、两角参考值及 5 秒脉冲。各进程使用独立输出目录，以免 ordinal 0 receipt 冲突。全部七张有两点实测增长后，才在 09:00.0 执行原 .1982 Native 的一次候选见证。探针专留四 CPU，七个保底子进程及控制进程需另有自己的 CPU 配额。候选和保底沿用同一有界 epoch。给启动、清场和第八张卡两点计数留下实际时间。
3. 探针归还时仍持有同一八卡 Slurm 步。先回收见证器和 hipprof 子进程，核对 PID/startticks、退出和原 PCI 上的项目持有者，再让原 PCI 上启动同一保底函数。其他七个子进程继续运行。确认第八张卡和其余七张的同期两点计数均增长后，才发布八卡恢复。探针退出未知时保留占有关系，禁止并发启动第八个进程。
4. 整个八卡步骤完成后，根在既有共享锁内消费实际 Slurm 终态和独立零卡 post，核同 boot、八 PCI、KFD 和项目设备 FD。只有该范围清洁后才续接普通八卡保底。换 Slurm 步存在清场和启动间隔，分别记录。同步骤内第八张卡复位无需第二个 Slurm 派发。

本包没有执行第 2–4 项。没有目标进程计数、加速卡忙碌时间、算子收益或训练结果。
