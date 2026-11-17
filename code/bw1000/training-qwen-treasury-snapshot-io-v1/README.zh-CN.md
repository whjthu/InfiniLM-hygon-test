# Treasury 快照 stdout/stderr 独立修复及 HOST-only 门禁

原 GPU-v2 request 前缀581bb322，pre步骤340/339/168均C0，但合并日志首行是srun限频告警，后续是唯一合法host_complete快照。原controller把pre.log整份json.loads而失败；0GPU及324missing必须保留，不能重播原v2。

修复以原v2 arm_controller.py SHA f961cb742ca0ea8b4ece43cf89df08d7b2d884d2dde817deb4aee75537240102 为精确基底。这里只交付小型零上下文unified patch（必须先核精确base SHA，不能对未知源宽松apply）与snapshot_io.py；不复制整个GPU包。patch为所有pre/gpu/post阶段分别新建phase.stdout/phase.stderr，保留完整字节，不筛告警。pre和post都调用read_host_snapshot，仅接受一份完整JSON对象且拒重复key/非有限值/附加JSON/非白名单schema，核实际accounting step、node、host_complete、完整FD、0holder、8distinctPCI及boot/monotonic锚点。stderr始终独立保存，不能拿它抽JSON洗掉告警。GPU控制器与后续独审器必须同步新文件名；旧目录和旧日志保持原件。

本包唯一可执行scope为三个串行0GPU HOST gate，base1087656/a01r4n14、old1087656/a01r4n16、domain1103606/f13r1n10，各2CPU/4GiB/0GPU；原epoch600s，Slurm最多120s整分钟向下floor，HOST snapshot进程最多60s+kill5，client与accounting同截止。独立stderr marker强制验证两条输出管道，调用真实snapshot.snapshot(False)；不进入容器、Torch、SDK、模型、权重或生成。三节点均需actual C0、strict HOST snapshot及独立root终态核验。任何失败或UNKNOWN不得重播，无GPU followon代码。

根普通发布、安装本清单exact源至 R/results/training-qwen-treasury-snapshot-io-v1/source 后，使用 /usr/local/python3.12/bin/python3 -B source/launch_once.py --publication COMMIT --claim-sha256 SHA --request-id NEW32HEX 唯一次调用。先核唯一namespace和label无原提交，父作业与node正确，根持有实际句柄。此CPU gate通过仍不是GPU已执行；下一GPU-v3必须新scope/newlabel/claim，接入实际CPU原件与新controller输出口径，原900/960/1800预算、数学和输入不变。
