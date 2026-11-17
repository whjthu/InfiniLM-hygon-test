# 三个独立单卡进程的 Treasury dev108 开发评测

一次root原epoch1800秒；base=1087656/a01r4n14、old_domain=1087656/a01r4n16、domain=1103606/f13r1n10，各1DCU/8CPU/65536MiB。三臂使用相同修订dev108，只有36signed_difference追加与训练相同FORMAT_INSTRUCTION；lookup/date72不变。greedy64、BF16、SDPA、batch1、4096上限等保留原native生成调用。没有DDP或跨parent collective。

根普通发布后只安装清单source文件至R/results/training-qwen-treasury-explicit-dev3-gpu-v1/source，其他数据与模型复用原固定目录。HOST绝对Python调用launch_once.py --publication COMMIT --claim-sha256 SHA --request-id 新32hex。只有一次原intent；三HOST控制器并行，各自pre/gpu/post，label含arm，保存PID/PGID/startticks，不启动SSHservice/poller。UNKNOWN保留原namespace/句柄，不创建fallback或重提。

先核330实际targetintent/outcome/log、源MF及13源、newinventory与实际accounting，不以publicsummary代替。新328按rootboundedauthority逐实际originalmetadata/witness/init/RNG以及checkpoint5必要文件hash核验，不重reader。旧285adapter及新328adapterconfig/weights/trainerstate各精确pin。每node实际provider版本与MODEL_ASSETS全部文件重新核验；完整modelhash在fresh45sPCI/KFD门禁和单可见HIP/leasedKFD资格之后、模型加载之前，计入900秒；避免长hash使ownershipsnapshot过期。每arm独立模型进程、唯一adapter，没有切换污染。

每臂加载/校验/生成hard最多900秒，coreutils timeout再kill10；实际GPU Slurm上限960秒整分钟，留给启动/退出。三臂DCU占用上限2880卡秒=0.8卡时，不叫busy或有效训练卡时。全链原epoch1800，GPU后保留180秒记账/post，所有实际Slurm分钟向下floor，client/query/wait/post按原剩余缩水。late core=min900,Slurm-60,remaining-180，保存实际workintent；不相等就不能宣称等预算胜利。pre/post各0GPU；post核全PCI/boot/KFD和实际workerPID退出。

每arm输出留R/arms/ARM/outputs，progress保存108ID计划和每条完成事件；任何失败/截断/缺失保全部324/每topic每arm36。根待原actualterminal和独立post后，以score_in_place.py --source 原Treasury冻结数据根 --output 新private目录，复用原_score(...,('base','old_domain','domain'))，domain明确新328，old_domain为旧285。所有三臂重新生成，旧298输出不复用。严格原grammar、不修末数；同时报告wrong/invalid_format/cap/empty/missing/invalid_response，所有分母保留。源码/CPU合格不是已执行，不做正式质量/盲测/性能或跨硬件因果宣称；旧训练实际roworder未独证。失败只能消费原件/修reader，不重Model，不续训或prompt/capgrid。
