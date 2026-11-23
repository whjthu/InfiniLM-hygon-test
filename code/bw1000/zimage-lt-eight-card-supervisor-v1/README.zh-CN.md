# 在同一个八卡步骤内验证 Z-Image 新算子路线

监督器先运行八个独立的 BF16 保底进程。子进程在第一次 GEMM 前核对精确 HIP 库、单卡 HIP 和 Torch 视图、物理 PCI 与 CPU 亲和性。八张卡的原 worker 脉冲均有两点计数增长后，仅停止 09:00.0 的子进程。其他七张卡继续运行。

探针运行前，再收集七张卡的两点增长证据。监督器同时核实 09:00.0 的原进程退出，以及 KFD 清单和项目 UID 下的设备 FD 只属于留下的七个保底进程。每个保底 PID 的 DRM render FD 必须映射到自己的卡。启动探针后检查七张卡脉冲新鲜度。探针和 hipprof 继承同一专用进程组，叶子第一次 Torch/Native 调用前再次核实实际 HIP count 和 PCI。监督器能追踪该进程组和被 Linux subreaper 接管的进程。清场有歧义时保留七张仍运行的保底卡至本步原预算结束。

见证器仍使用原 .1982 DSO，原 M288/N3840/K3840、两输入布局、32 MiB 工作区、候选排除及单次 BF16 数值和 trace 校验。此范围只产出算子身份和正确性证据。七张卡共驻留，正式性能比较调用为零。候选无新路线或数值不合格时，只要退出和持有者全部闭合，同一步仍恢复第八张保底卡。

探针结束后，重新核对 KFD 和 DRM 持有者，再在同一 Slurm 步和 09:00.0 上起新的 BF16 进程。再次见到全八卡各两点增长后，继续计算到同一 epoch 内预留的协作退出边界。记录进程组的受控退出和私有回执。根协调者在现有共享写锁下独立核对实际 Slurm 终态和零卡清场，再正常续接后续八卡步骤。换步骤间隔、09:00.0 在单步内的两次迁移及其计算证据分开记录。

## 本地渲染和设备入口

    python3 -I -B prepare.py NEW_RUNTIME_DIRECTORY

渲染器验证原源码和 v3 两个已发布运行时 SHA，复制原 BF16 worker 和 HIP 准入模块。它仅修订见证器的唯一 request、标签、处理器和掩码边界、进程组继承。测试逐个比较关键数值函数的 AST，并比较原 contract.py 字节。产物包括七个精确文件，设备入口强校验完整运行时 SHA。

    python3 -B -u /pair-hybrid/supervisor.py --scope /pair-run/scope.json --scope-sha256 SHA --runtime-sha256 SHA --out /pair-run/supervised

此入口由根在新八卡同节点单任务步骤下调用。绑定已验证 DTK 镜像、原 .1982 Native 到 /pair-build、原 /training-runtime/site、原 HIP 工具和同机 hyhal 路径。运行时目录只读，私有输出目录可写且必须是新命名空间。两条已验证旧入口的 hyhal 源路径写法不同，根需先通过该节点实际 canonical 来源核对其兼容方式。保留原 --rocm、Python site 和 HIP/RCCL 环境，保持与目标 KFD PID 可对应的进程命名空间。固定父作业 1103606 和节点 f12r2n11，实际要求八卡和恰好十二个 Slurm 绑定 CPU。七个保底 CPU 与探针四个 CPU 不重合，目标卡恢复后复用探针 CPU 中的一个。另一个 CPU 用于入口和监督。.2075 拒绝进入该入口。

scope 绑定唯一 request、label、parent、node、fresh_root_admitted、boot_id_SHA256、allocated_DCU、PCI 八项、independent_pre_step、independent_pre_SHA256、epoch 和 deadline_epoch。根先以实际资源账、当前清场证据、唯一派发记录和原生库校验制作 scope。首尾跨度恰为 240 秒，子进程窗口只从这一个期限扣减。内存、容器挂载和外部独立验收按根的实际 8 卡 scope 另核。

本机检查使用模拟进程和模拟设备持有者。该版本尚待目标验证 12 CPU 亲和性、真实 HIP 掩码继承、每个 Torch 子进程的实际单卡 render FD，以及 leased KFD 的可见范围。掩码只提供 SDK 可见性，物理持有者另行核对。实现对额外或不可读持有者停止接手，遇到 EBUSY 或 EPERM 保留失败。
