# GLM fresh K/V Native 编译 HOST 控制器

原请求 e5ead062820a840906c74fce7294023b 对应已发布 worker、recipe 和 claim（34da8adfefc5f08d3ceef01b462d546e7d2b9c69）。固定 O01 1103606/f12r2n13、4 CPU、16384 MiB、0 GRES。Native 只复用已有 gfx936 对象，计划 2 次 fresh M、2 次 HOST 编译、30 个原对象复用和一次 33 对象链接。

    /usr/local/python3.12/bin/python3 -I -S -B host_once.py

入口先调用已发布的原 kernel 控制器只读核对请求 b9a7f0b78c566911ed40004fc906ce44。实际 C0、独立对账及原对象哈希齐备后，另观测两个已知父作业，确认该原 numbered step 已消失，指定节点无新步。这份记录覆盖原零 GRES 作业的 Slurm 占用释放。只有上述证据完备才开始 Native 的单 epoch 和单独预检。两个父作业的实时所有权、O01 节点、父结束时间及原标签唯一性均须符合。

使用冻结的 CPU308 DTK ENV 和 Apptainer 环境，保留原 ROCm provider 挂载，运行时设备掩码全部为 -1，Slurm 明确 --gres=none。指定 --cpu-bind=threads，并在引导器读取并记录当前进程的实际 CPU 集合。只有恰好四个 CPU 时才载入原 worker，终态对账核查同一步号及记录 SHA。CPU308 的历史启动未记录 CPU mask。作为明确挂载的 checkout、原 gfx936 对象及原 kernel 对账文件均为只读。由隔离 Python 引导器显式载入 checkout 内的原 runner，保证同目录模块可见。原预算为 epoch+1800 和 epoch+1680，Slurm 时限按同一时钟扣除已耗时间。只提交一次，预先固定原 intent 和 exec-attempt 记录。

    /usr/local/python3.12/bin/python3 -I -S -B host_once.py --reconcile

只读模式核对原证明和时钟，读取一次原 step 记账。未见终态则保留 UNKNOWN。同一成功 step 要同时通过 2M/2HOST/1link、owned child 已回收、原 kernel 绑定、实际 M/MD 集、强导出及 provider 的 worker 证据、共享 ELF 头和产物 SHA。输出保留终态行、worker、recipe、四个依赖文件、ELF 的 SHA，并保留原 controller 结果。此入口不写入队列，不释放预约。后续资源释放依赖单独的实际 post 清理。

本机夹具覆盖原先验、成功、UNKNOWN 后原步恢复对账、错归属、父余时、重复标签与实际 MD 不一致。使用从正式容器命令抽出的隔离引导器，实跑原 worker 的 --help 验证同目录导入。Slurm 与目标 worker 构建结果均由本机夹具提供。
