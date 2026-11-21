# T09 单DCU普通缓存训练：独立功能回归

v2 独立 source-only 交付；root 已明确授权把本次执行权限写入实际CLAIM：root_GO/GPU_GO/training_GO=true。此agent仍不连接/派发，root须发布与fresh placement后唯一调用。原 cache 的既有授权范围、原 Transformer 和共同393入口不变，普通 common CLI 执行原 Muon/r4-alpha8/GC/BF16 配方。25次实际更新=5次计时预热+20次计量，原 `validate_recovery=false`；没有额外恢复调用。一个 cache 的重复曝光不等于25个独立视频。原首步 replay、UNKNOWN transfer 与正式数值/效果/性能欠账均保留；不把这个候选称为新增公开域或正式17通过。

固定 request `f136efa75c0b480a9055375a0a632f08`，RUN=`/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-cached-native25-target-v2`，仅允许父1087656/f11r4n17。root先核无队列/owner hold和当前资源，再独占新RUN，放源码到RUN/src，放固定SHA的preflight到RUN/preflight.json。原目录/请求绝不重播。本包没有默认SSH、下载器或后台服务。

**路径纠正：**此前本机 fallback plan 把HOST source误写成 `/source/hunyuanvideo-1.5`，历史结果不改。原 scoped38源实际直接位于HOST `results/training-hunyuanvideo-original-source-v1/source`。本包严格按旧审计 command 将该HOST目录只读bind到 `/t09-native/hunyuanvideo-1.5`；普通common资产映射使用这个container alias，source-root=/t09-native。NATIVE_BINDING.json是原26f1c4…精确副本，逐38成员稳定stat+bytes/SHA核，不写原source。

原权重historical st_dev47、login54、compute49不可混用。新scope明确固定preflight.model.host_file_identity=[54,11025132338,33306632192,1794715911000000000,1794715912000000000]和file_identity=[49,同一后四项]。HOST controller显式execution_mount=False核完整HOST54；GPUworker显式True核完整compute49；两种观察不可互换，47仅历史证据。必须verification=EXECUTION_MOUNT_FULL_SHA256并绑定root实际完整33,306,632,192B SHA重发现收据；worker在实际mount重新核完整5tuple，绝不忽略st_dev。full_SHA_receipt{path,bytes,SHA256}硬绑定实际.733 normalized收据932B/8311536400438e4aac50a361b89030633d952ebb5b320ad4e9771b8d3759aff4，并核node/path/size/SHA/全部fingerprint/stable_stat/payload_bytes_read。

preflight合同见contract.validate_preflight：state=T09_CACHED_NATIVE_TARGET_PREFLIGHT_PASS、passed/root_GO=true、固定request/parent/node/run、原epoch和epoch+1440；model精确身份；cache3521630B/9bdb…及config968B/689b…；framework_manifest=e54fd…；native_source_manifest指RUN/src/NATIVE_BINDING.json；runtime_manifest指既有 `results/training-t06-sdxl-runtime-v1/receipt.json`（2124B/56f65e63…）。模型真实config再核同968B/689b…；完整393在GPUworker内重新核。runtime receipt是既有资格，不冒本次完整native provider已成功。

本次root明确授权的root_GO/GPU_GO/training_GO已经为true，不能再只在包外task写授权。READ_GO也由root明确预注册为true，仅实际Slurm终态后消费原件；所有实效flags由同一manifest闭合，launch和terminal reader均逐项拒绝不一致；Model/Dataset/formal17/效果/性能不因此晋级。root通过原通道调用：

```
/usr/local/python3.12/bin/python3 -B RUN/src/controller.py --workspace RUN --epoch ORIGINAL_EPOCH --source-manifest-sha256 FINAL_MF --preflight-sha256 EXACT_PREFLIGHT_SHA
```

origin须由root整个实际请求最初捕获并贯穿源码验证/asset核验/部署/dispatch；controller和worker不会重置。GPU最多1200秒且实际srun时限、client等待随1440原epoch缩水；work截止origin+1240，保post/accounting。pre/post各0GRES、1CPU/1GiB，GPU为1DCU/8CPU/128000M。query只原parent，不静默丢accounting坏行。每phase先落intent再Popen；timeout只强身份匹配的自有srun进程组，终态不明不新step/replay。GPU只有实际terminal后才独立post。自有client reaped+真实Slurm终态+独立post是本范围释放证据，不宣称全机器排他。

HIP实际用原runtime_pci/assigned_kfd绑定Slurm单卡和PCI；主运行走common `--accelerator hip`，Torch的cuda命名是HIP兼容API，不推断NVIDIA。原systemalias/APPTAINER/DTKENV沿旧审计GPU挂载；无未经实证的全8KFD访问要求。原snapshot权限/未知不改为0。

真实输出在RUN/outputs/hunyuanvideo-1.5/train/本label。导出审核要求实际1..25 steps.ndjson有限loss/非零梯度/持续正时间、Muon1769472参数与原配置、adapter确实改变、HIP completed manifest，10/20/25保存state和final adapter的实际SHA匹配manifest。记录在原native optimizer.step完成后写盘，源码不伪造。失败保25全分母及已读前缀计数；缺失/错字段不算PASS。独立read_terminal.py只再核这些原件、原scope/source/preflight/actual accounting/post，输出≤16KiB，不重模型/训练/反序列化PT。checkpoint hash会读取已保存state文件正文，实际量不假称0；backend本身还会加载并hash模型，只有入口stat证据是0权重body。

本机check_CPU四类为synthetic：ingress、late/expired实际argv、7列accounting拒错、stored-record/export分母。Windows最初3次夹具stat/fstat失败保留；最终parser测试注入实际小文件bytes/hash隔离POSIX原子reader，不放宽生产reader，不据此冒目标身份/模型/数值资格。无Torch/GPU/集群实际调用。

## v1 原prelaunch保留

v1 staging已存在，原controller/Slurm未启动；root实读原固定labels全部缺失、preflight/intent缺失，认证返回的本机错误指向verify_package中的operative claim门禁。v1三执行flags为false，与包外task授权不一致。该原件不改、不重放、不推断任何GPU更新。本v2仅新namespace/request与实效claim，原source/预算/数学/双mount保持；新增check_package_CPU.py必须用真实v2 manifest过正向，再对重新封装的false operative claim拒绝。若Windows stat/fstat接口差异，仅测试注入Windows fstat元数据，实际全部文件bytes/SHA与manifest仍真实读取。
