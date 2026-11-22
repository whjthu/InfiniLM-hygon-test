# InternVL 原 COCO64 八卡首两步诊断

目标采用 1087656/a14r3n04 的完整 8 DCU、64 CPU 和 224000 MiB。保持原 COCO64、Swift 原生 InternVL HF 模板、BF16、SDPA、max_length 4096、梯度检查点、每卡 batch 1、累积 4 次、有效 batch 32、LoRA 16/32/0.05，以及原 AdamW/cosine 25 步调度。Swift 在第 2 次真实 optimizer 更新完成并通过全部记录边界后正常停止。8 rank 每 rank 8 个真实前向与 2 次真实梯度裁剪和更新。捕获输出保存指纹、统计和状态变化，原始激活或梯度正文不随捕获复制。

程序复用实际 .1004 的 compute 模型挂载证据，并在当前新编号步骤重核 20 个共享文件绑定，再在 Apptainer 内核对 /internvl-model 下 17 个标准成员都是原 inode 的只读 regular 文件。校验 .360 实际得到的 504 个 LoRA 名称、形状、类型与梯度检查点 provider。经 .362 传入的 rank 0 初始化文件只装载一次，使用原 ordered 参数 SHA 校验全部值，经八进程广播后各 rank 再核同一 SHA。按 .362 的 64 个原实际 post-model 输入指纹核对每个 rank/微批次，应用已部署的跨 rank 输入守门。

本包使用新的实验标识。原 1087656.1008 在 GPU 工作启动前因选择器字面顺序检查失败。复用统一框架的 target_signature 比较字面候选层顺序，仍逐项核对整个有序参数清单。HOST 控制器在申请 GPU 前调用同一 native.binding。由项目已有 HOST 单次派发器创建本机请求、实施资源交接、保存一份 a14r3n04 同 boot 的独立全 FD/PCI/KFD 预检，随后仅执行一次新编号 step。GPU worker 固定参数如下。

```text
node.py --epoch <原单epoch> --claim-sha256 <采用claim SHA> --manifest-sha256 <采用清单SHA>
        --exports-sha256 <原 .1004 export-pins.json SHA>
        --before <GPU前置原JSON绝对路径> --before-sha256 <前置原SHA>
```

前置入口复用本包 snapshot.py，设 NODE=a14r3n04 后调用 snapshot(False)。GPU worker 要求同 node、boot、8 PCI，两个观测均无 KFD 进程，跨度不超过 45 秒。Slurm 标签为 training-internvl-native64-first2-hip-v4-gpu，步骤从 epoch 算最多 1500 秒，Native 工作最多 1440 秒，外层整请求最多 1800 秒。RUN 为独占新目录，由 node.py 创建。HOST 前置收据放在 RUN 之外。独立终态读取各 rank 的 terminal.json、实际 Slurm 记账及更晚同父 PCI/KFD/PID 清场。

原 A100 其余 rank 的 RNG 和完整参考值未到目标。本包保持八个 HIP rank 的原生新 RNG。当前资格范围是同输入、原 LoRA 初值和真实前两次更新的工程诊断。正式数值对齐、连续 25 步、恢复、A100 性能和留出效果仍待对应验证。当前只进行了本机源码与小型替身 fixture 检查。

实际 1087656.1017 测得目标容器 Transformers 5.5.0，InternVL 与 Qwen3 数学源码分别与原 .360 SHA 一致。A100 参考仍为 5.12.1。v3 以同一 runtime_identity 在零 GRES 容器中核对版本及源码，在 GPU 训练前再次核对。训练中仍使用原回调复核实际模型 provider。

原 1087656.1029 在单卡同容器中证实显式私有 /tmp 消除了可复现的 LLVM ENOSPC。v4 在八卡工作中把本次 RUN/tmp 挂载到 /tmp，容器内核同一个目录身份和实测至少 1 GiB 可用空间后才初始化设备。GPU 前的零 GRES 预检也走同一目录校验。
