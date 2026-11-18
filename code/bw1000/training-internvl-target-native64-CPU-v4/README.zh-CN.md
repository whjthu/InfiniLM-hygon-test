# CPU-v4：明确目标平台源码契约

这是新的一次 CPU 候选，全部 GO=false，需 root 独立审查和发布。保留357真实失败，不修改原v3或共享运行时。

保留原 NATIVE_BINDING.json 字节及 A100 源码 pin。TARGET_MATH_SOURCES.json 单独登记已取回的目标 InternVL 632fc3b… 与相同 Qwen3 fbdcfe…、公开差分证据 e1cdefc6…；不是临时接受任意观测 hash。

实际 provider 先枚举 module/class、module.__file__、inspect 源路径、resolved 路径与 bytes/SHA，完整观察在校验前持久保存。枚举失败也保存已有记录和错误摘要。measured binding 副本只显式使用目标 provider 契约；504名称/形状、43646976、GC/freeze、原64输入、4096、EOS/logit选择均保持。

CPU最多300秒同origin，work210/Slurm240/原清理预算，8CPU12GiB0GRES、8GiB采样RSS不是硬上限；原绝对Apptainer/只读hyhal、三visibility=-1、lazy-init拒绝及held/tree清理保持。implicit library GPU行为仍UNKNOWN。

预置consumer从实际请求context与安装MF/claim绑定，核观察与meta结构、原64分母及两层终态。跨login PID不可核则UNKNOWN，不授予GPU释放。CPU通过不证明模型forward数学、初参、跨平台RNG、first-update或训练数值等价。

仅 `python -B verify_package.py --prepared` 检查准备态。launcher/receiver调用默认verify，要求root显式采用；--prepared不会解锁远端执行。禁止自动retry或用新label重放UNKNOWN。
