# Treasury 三节点零GPU容量/缓存 HOST gate

由原GPU-v3 base实际LLVM ENOSPC触发。这里不假设64MiB或任何路径已是根因，不准备GPU-v4，不清理任何旧文件。

一个新namespace、单epoch300秒，base1087656/a01r4n14、old1087656/a01r4n16、domain1103606/f13r1n10，三个串行Slurm step，各2CPU/4GiB/0GRES/60秒；worker50+kill5，accounting/client都用原剩余预算，余量不足90秒不新开stage。每stage做HOST和原容器对照。原v3 flags/binds严格复用：--contain --rocm --no-home --pwd /，原RUN/源/模型/输入/media/runtime/hyhal/adapter路径及模式完全相同。新probe由stdin送入，不新增挂载；HOST stage日志仅自己新scope写入。容器使用-I -S标准库Python，不importTorch/SDK/LLVM、不编译、不读模型权重或hash权重。

固定路径仅 /、/tmp、/dev/shm、原v3 RUN、新scope RUN，以及白名单环境路径：HOME/TMPDIR/TMP/TEMP/XDG_CACHE_HOME/TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR/MIOPEN_USER_DB_PATH/MIOPEN_CUSTOM_CACHE_DIR/MIOPEN_CACHE_DIR/AMD_COMGR_CACHE_DIR/LLVM_PROFILE_FILE/APPTAINER_TMPDIR/APPTAINER_CACHEDIR。只读路径值；URL、凭据赋值和非绝对路径只记录拒绝及SHA。HOME下少数cache目录明确为候选默认路径，未调用库证明其实际使用。不存在的路径最多向上32级找existing ancestor，并明确标注。

对这些路径只做statvfs和一次批量/usr/bin/df -i -P；从/proc/self/mountinfo单一元数据文件筛选最长匹配挂载，仅记fstype/major:minor/mountpoint/free/total/inodes，完全不输出mount source/server/addr。记录uid/euid/gid/cwd/sysflags。两个精确apptainer.conf候选只hash整个小文件并只输出sessiondir max size整数值（单位另未独证）；不改配置，active config未知时明确未证。sysflags、HOST与container容量差异是诊断证据，不能自动定位实际LLVM写入路径。

不写canary，不清理/改共享tmp/cache，不启动service/poller；--rocm和容器运行时自身隐含探测/会话管理不冒称为显式SDK调用。actual3stage终态、source/payload/双流/同epoch和容量数值均待根普通发布后唯一执行及独审。GPU_GO始终false，容量可见也不准直接GPU；若不足只报告，不删除未知文件。

源安装到R/results/training-qwen-treasury-capacity-host-gate-v1/source后，HOST绝对Python调用launch_once.py --publication COMMIT --claim-sha256 SHA --request-id NEW32HEX一次。已有namespace/label或未知执行只reconcile、不重放。原V1/V2/V3失败和所有分母不变。
