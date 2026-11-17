# 自有缓存与真正HOST-main的唯一未来CPU scope

精确source/11文件保持candidate MF c491不变。执行namespace仍R/results/training-qwen-treasury-cache-host-main-candidate-v1，source/放原11源，cpu/放本CPU控制，根目录CPU_CLAIM.json/CPU_PACKAGE_FILES.json。新Slurm label training-qwen-treasury-cache-host-main-CPU-v1-(arm)，未安装/未提交/无实际cache写入。

三T01原节点串行：1087656/a01r4n14、1087656/a01r4n16、1103606/f13r1n10，各8CPU/12GiB/0GRES；同原900epoch。每step Slurm最多180秒整分钟floor，work最多150+kill10，late按原剩余缩水，不够120step+60reserve拒新stage。client/query/accounting同deadline且自己的terminal窗口。首CPU失败或UNKNOWN停止后续派发，仍保3arm/324InferRequest/120KiB计划分母。原namespace或label已在则只reconcile，不重放。

实际container flags固定 --contain --rocm --no-home --pwd /，脚本内部cd自有RUN；原candidate文字中--pwd新RUN的未来建议由本CPU claim明确覆盖，11源保持exactbytes。ownRUN精确RW、source RO、model/旧330dev和media nested RO、Swift/vision/minicpm/hyhal RO。为CPU构造两个AdapterRequest，旧285/新328目录各RO给三arm；没有Engine实例、load_model=True、推理或训练。HOST imports后的库内部device查询/JIT副作用UNKNOWN，visibility由childshell真实置空而非mock CUDA成功。

childshell在Torch/Swift前设TMP/XDG/TorchExt/Triton/Inductor/Comgr/MIOpen及HOME/HF_HOME/HF_MODULES_CACHE到ownedRUN，离线且不copy旧home或凭据。每目录canary4KiB×10=40KiB、三臂最多120KiB，fsync/read/hash后只删新建且inode一致的owncanary。只证明测试字节可写；userquota/任意JIT容量均不合格，库import实际owncache写入总量UNKNOWN。真实CPUprocessor(load_model=False)/template/API signature不替代GPU数值或训练。GPU_GO永远false，绝无GPU-v4 followon。

Slurm stdout stage.stdout与Slurm.stderr独立；container内canary/main stdout在canary-hostmain.stdout，DTK startup/Swift/native stderr完整保container.stderr。资格读取独立owned-cache-readiness.json与host-main/host-main-receipt.json，不把整份日志当JSON。

旧330 actualC0/原source13/inventory07为运行前硬前提；capacity64MiB事实只有producer reconcile c008，不是独立PASS，旧独审UNKNOWN不重播。timeseries53待授权数据CPU域不访问。根ordinarypub和freshparent/node去重后，以HOST绝对Python运行cpu/launch_once.py --publication COMMIT --claim-sha256 CPUCLAIMSHA --request-id NEW32HEX一次；所有资源/写入仅该CPU scope，不含GPU。

真实container前与结束/失败后，同CPUstep各单次ownership-pre/post：仅当前server的8DRM PCI、boot、自己UID/PID/PGID/startticks、KFDproc PID列表与同UID FD指向GPU节点的持有者，不读任何/proc cmdline/environ。权限/未知/超预算均拒绝，持有者必须0；比较pre/post PCI/boot和stagePID/startticks，原ownedcontainer PID必须退出或身份已不相同。snapshot_io仅为双流helper，真实inventory来自新ownership.py，不冒继承旧门禁。pre/post消耗同150work，container主流程按剩余减20秒（10kill+10post）缩水，900epoch不重置；freshness不是busy测量。
