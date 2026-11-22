# InternVL 原 COCO64 八卡首两步诊断

目标采用 1087656/a14r3n04 的完整 8 DCU、64 CPU 和 224000 MiB。保持原 COCO64、Swift 原生 InternVL HF 模板、BF16、SDPA、max_length 4096、梯度检查点、每卡 batch 1、累积 4 次、有效 batch 32、LoRA 16/32/0.05，以及原 AdamW/cosine 25 步调度。Swift 在第 2 次真实 optimizer 更新完成并通过全部记录边界后正常停止。8 rank 每 rank 8 个真实前向与 2 次真实梯度裁剪和更新。捕获输出保存指纹、统计和状态变化，原始激活或梯度正文不随捕获复制。

程序复用实际 .1004 的 compute 模型挂载证据，并在当前新编号步骤重核 20 个共享文件绑定，再在 Apptainer 内核对 /internvl-model 下 17 个标准成员都是原 inode 的只读 regular 文件。校验 .360 实际得到的 504 个 LoRA 名称、形状、类型与梯度检查点 provider。经 .362 传入的 rank 0 初始化文件只装载一次，使用原 ordered 参数 SHA 校验全部值，经八进程广播后各 rank 再核同一 SHA。按 .362 的 64 个原实际 post-model 输入指纹核对每个 rank/微批次，应用已部署的跨 rank 输入守门。

根采用本包时将 CLAIM 的 root_GO/GPU_GO 设为 true，再封清单。由项目已有 HOST 单次派发器创建本机请求、实施资源交接、保存一份 a14r3n04 同 boot 的独立全 FD/PCI/KFD 预检，随后仅执行一次新编号 step。GPU worker 固定参数如下。

```text
node.py --epoch <原单epoch> --claim-sha256 <采用claim SHA> --manifest-sha256 <采用清单SHA>
        --exports-sha256 <原 .1004 export-pins.json SHA>
        --before <GPU前置原JSON绝对路径> --before-sha256 <前置原SHA>
```

前置入口复用本包 snapshot.py，设 NODE=a14r3n04 后调用 snapshot(False)。GPU worker 要求同 node、boot、8 PCI，两个观测均无 KFD 进程，跨度不超过 45 秒。Slurm 标签为 training-internvl-native64-first2-hip-v1-gpu，步骤从 epoch 算最多 1500 秒，Native 工作最多 1440 秒，外层整请求最多 1800 秒。RUN 为独占新目录，由 node.py 创建。HOST 前置收据放在 RUN 之外。独立终态读取各 rank 的 terminal.json、实际 Slurm 记账及更晚同父 PCI/KFD/PID 清场。

原 A100 其余 rank 的 RNG 和完整参考值未到目标。本包保持八个 HIP rank 的原生新 RNG。当前资格范围是同输入、原 LoRA 初值和真实前两次更新的工程诊断。正式数值对齐、连续 25 步、恢复、A100 性能和留出效果仍待对应验证。当前只进行了本机源码与小型替身 fixture 检查。
