# GLM fresh K/V 的单次 Native CPU worker

`tools/run_glm474_fresh_kv_native_once.py` 在已准入的 O01 CPU step 中执行。原认领固定 `1103606/f12r2n13`、4 CPU、16 GiB、0 GRES、1800 秒。所有准备、依赖验证、编译和链接均使用同一个 epoch，工作截止为该 epoch 后 1680 秒。原始 kernel 任务必须已有目标编译终态及独立清理证据。

执行顺序固定为两次 fresh M、两份依赖复核、两次 HOST 编译、两份实际 MD 复核、沿用对象重新核验、单次链接、ELF 与库依赖审计。新 Native 使用独立目录，存在时对账原运行。命令在自己持有的进程组运行并沿用原 CPU308 的按 PID 和进程组清理逻辑。结果保留实际步骤、已尝试和已完成计数、失败阶段和最终库 SHA。

当前已实现 worker 和固定 CPU 范围，并验证本机成功顺序以及依赖、对象、链接和资源失败边界。HOST 控制器负责目标资源准入、Slurm 提交和独立终态对账。该 CPU 步不计 GPU 利用率、训练队列或推理性能。
