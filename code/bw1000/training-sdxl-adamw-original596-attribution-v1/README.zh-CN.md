# SDXL 原 1087656.596 首步更新的 CPU 归因

原 GPU 1087656.596 与独立比较 1087656.598 已完成。官方 1120 张量、5,806,080 元素的更新相对 L2 为 19.3993%，门槛是 5%。裁剪前和裁剪后的梯度汇总均为 2.2963%，汇总相同仍需从各臂档案核验逐元素的 BF16 位模式。原 1087656.398 使用更早的 176 档案，提供了 CPU 复算方法和反号诊断依据。

新诊断读取本次原 GPU 归档的初始参数、裁剪前后 BF16 梯度、保存的范数、优化器组成与步数、实际最终参数。以 A100 原生更新平方范数作为统一分母，将原生更新差分成裁剪前梯度差的影响、裁剪带来的增量、各平台原生与同梯度 CPU 更新残差的差。保留三项的有符号交叉项，要求复算原生总差及其分母与 .598 一致。若两臂的裁剪均逐位恒等，复用裁剪后的 CPU 结果。另为两个都非零而反号的位置，只将 Hygon 的符号对齐 A100，保持其幅值，执行一次解释性的 CPU 步。输出这条假设的残留误差，也记录 A100 幅值大于 16 倍 epsilon 的反号位置及占原生平方差的比例。达到 5% 门槛需要消除 93.3569% 的原生平方差。

## 只读入口

已有 GPU 原件位于 R/results/training-sdxl-sdpa-adjoint-fp32-v1-a93d41a0d31140f1ad99bb105a024e99/。入口使用原 terminal、1,783,928 字节的原 comparator 和 935 字节的原 summary 摘要链锁定该目录下 73,890,269 字节的 candidate/replay/observed.pt。参考目录为 R/results/training-t06-sdxl-first-reference-r2/，使用 2,677 字节的原 policy 和 74,257,307 字节的 A100 reference.pt。两份归档逐个通过现测大小与 SHA 后，才使用 CPU、weights_only 和 mmap 解析，逐张量检查 BF16、形状、有限性及顺序。原始输入图像、模型权重和归档中的其他存储无需载入计算。输出只有固定大小的 JSON 指标。

复用 .598 的 CPU 容器与 Torch 2.10.0。建议 0 GRES、4 CPU、8192 MiB、Slurm 120 秒、进程上限 100 秒。历史同类复算使用 4 CPU/8 GiB。本轮多出一次方向性复算，每份额外 BF16 参数数组约 11.1 MiB，实际峰值待测。节点候选 1087656/f12r3n20 需 root 新鲜资源准入。输入目录按只读挂载，容器复用 .598 的冻结离线依赖。CPU-only 环境将 CUDA_VISIBLE_DEVICES、HIP_VISIBLE_DEVICES、ROCR_VISIBLE_DEVICES 设为 -1。

```sh
/usr/bin/timeout --kill-after=5s 100s /usr/bin/python3 -B -u /audit/read_original.py \
  --run /original --public /audit/original596-public.json --reference-dir /reference \
  --optimizer-helper /audit/adamw398-worker.py
```

原件摘要对应仓库的 training-sdxl-sdpa-adjoint-original596-numeric-failure.json，旧 CPU helper 对应 training-sdxl-adamw-first-step-counterfactual-v1/worker.py，每份都按既有摘要固定。当前完成本机源码与边界验证，原私有档案的归因结果仍待目标 CPU 实测。正式训练验收维持失败。
