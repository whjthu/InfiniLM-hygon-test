# MiniCPM checkpoint-20 原生首个前向见证

原 1103606.1171 的恢复与连续分支在 step 20 参数、优化器、调度器、存档和恢复时 RNG、160 个成对输入指纹上相同。首次差异出现在下一次训练更新之前。首微批覆盖原先在该处已不同的六个 rank，rank 1 和 6 的原首次差异较晚。原连续分支没有模型入口瞬间的 RNG、实际模式、autocast 和 SDPA 算子记录，新的恢复记录无法补回历史连续记录。

新入口保留实际 checkpoint-20 恢复、原 100→80→20 跳数、八 rank、梯度累积 4、BF16、原 SDPA 及 max_steps=25 配方。原恢复回调已记录 native RNG loader 的返回状态。新增边界记录 compute_loss 进入、模型根 forward 进入及返回后的 RNG，模型整树 training 指纹、各 Dropout 模块设置，根、语言模型和首个 Linear 的实际有效 autocast，以及 CPU profiler 观察到的实际 SDPA 和 dropout 算子计数。看不到底层算子时保留未知。所有观测仅保存摘要，profiler 能覆盖的算子名、模块配置和 RNG 指纹均有各自的证据边界。

## 安装与启动

将 native_observer.py、probe.py 放在同一个新的只读插件目录。使用原 1171 已验证的 Swift effective command，保留原八 rank 环境、框架、模型、数据及插件顺序。把原 minicpm_usgs_witness.py 的单个 external_plugins 路径改为新目录的 native_observer.py，diagnostic callback key 仍为 minicpm_usgs_resume_witness。新插件按原 SHA 和字节数载入旧见证器及同目录 update_contract.py。旧框架及源码目录留在原挂载，checkpoint 参数使用原 continuous/train/checkpoint-20 的只读实际路径，输出与进度位于新私有目录。更新 MINICPM_USGS_CONFIG 与其 SHA，使它明确绑定原 checkpoint 和完整 checkpoint_files，同时将 MINICPM_USGS_EPOCH、DEADLINE、MAX_SECONDS 绑定到新单一预算。额外设置：

```sh
export MINICPM_FIRST_FORWARD_FROZEN_WITNESS="$ORIGINAL_SCOPE/src/minicpm_usgs_witness.py"
# 由现有 launcher 执行新配置下的原 effective command。实际 argv 中使用：
# --external_plugins ... "$AUDIT_STAGE/native_observer.py" ...
# --callbacks minicpm_usgs_resume_witness
# --resume_from_checkpoint "$ORIGINAL_SCOPE/continuous/train/checkpoint-20"
# --output_dir "$NEW_SCOPE/probe/train"
```

以原 Trainer.compute_loss 的真实回调为停止边界。原 DDP/model forward 和标量 loss 已返回后，每 rank 写本地证据、同步自己的 CUDA 流，再原子发布与同一 config、epoch 和 Slurm step 绑定的 ready。八个标记都可见后，每 rank 写 stop 记录并以 SystemExit(0) 正常退出，Trainer 不再进入 backward、optimizer 或保存。缺 rank 最多等待 45 秒并按原绝对截止进一步缩短；该栅栏使用有界文件检查，不增加阻塞 collective。原 GPU forward 或 CUDA synchronize 的 native 故障仍由实际 launcher 的有界等待和受控进程回收处理。正式启动需新独立 claim、fresh PCI/KFD/端口、同配方资源绑定与实际终态/清场。原 .1171 controller 的 25+5 完整窗口判定也需更换为本次八 rank stop 判定。

## 比对

```sh
python3 -B "$AUDIT_STAGE/compare_first.py" \
  --original-root "$ORIGINAL_SCOPE" --original-public "$ORIGINAL_PUBLIC_RECEIPT" \
  --probe-train "$NEW_SCOPE/probe/train"
```

比较器按 .1171 已独立校验的 paired-recovery.json 和原文件 SHA 读每 rank 的原首输入、状态和两个标量，并核对所有新 rank 的停止记录。可用 --repeat-probe-train 提供另一份相同前提的新快照，逐项比对 RNG、模式、精度、算子和标量。原连续阶段只提供历史首标量，没有历史运行时元数据。结果用于决定下一定位边界，正式恢复验收维持失败。
