# MiniCPM Checkpoint 20 原生 RNG 首前向对照

`1103606.1189` 的 8 个 rank 都走同一条 Flash SDPA 路由，每 rank CPU profiler 记录 839 次选中算子与 186 次 native dropout。八 rank 从原生 RNG 恢复到模型 root 前一致，root 至 loss 均有 RNG 变化。首微批次 loss 只有 1 个 rank 与 `.1171` 的历史 resumed 结果完全一致。历史没有同一 RNG 边界的记录。

第二次调用同一 Trainer/DDP 会经过已经运行过一次前向的 reducer、模块缓存、Trainer 包装和 AMP 运行时。已有源码无法完整恢复并独立核对这些状态。本试验采用同一节点和同一组实际 PCI 的三个新进程组，顺序为 **A1 → B → A2**。每组八 rank、每 rank 仅运行原首微批次一次，并在反向前退出。各组以相同字节的原 checkpoint、数据、recipe、原恢复路径和绝对 epoch / deadline 启动。A1 与 A2 都对本 rank 已还原的默认 HIP generator 重新设置其原字节。B 使用私有 generator 和固定种子 `0x4D43504D3230 + rank` 生成替换状态，仅设置本 rank 当前默认 HIP generator。测量点位于已核一致的原生 RNG 恢复后、原 `Trainer.compute_loss` 前，逐项要求 Python、NumPy、Torch CPU 与同进程其他七卡默认 RNG 保持一致。入 root 再验证应用后的完整 RNG 状态。

原插件继续核对首微批次完整输入指纹、Hugging Face/Swift 原生累积游标、LoRA/优化器/调度器恢复态、空梯度及每 rank 的 DDP/autocast/SDPA/dropout 路径。新增探针对同一前向观测 root 参数版本和输入对象、形状、类型、stride 及版本。模型注册 buffer、选定 rotary 属性和可观测缓存属性单独读取并保存摘要，显式记录不透明缓存属性。观测上限为每次 64 MiB，不复制整套底座权重。参数版本和输入的结构校验针对 PyTorch 可观测变动。设备原生私有状态、opaque cache 与实现绕过版本计数器的写入仍有盲区。缓存变化可以是真实前向的一部分，以证据保留，不在进程间继承。

聚合器保留 24 次计划前向及全部缺失/拒绝。A1/A2 的逐 rank loss、前向末 RNG 和可观测末状态先做一致性判断。若 A 对照本身有差异，该轮记录同配方新进程的非复现。若 A 对照一致、B 的 loss 与两 A 均不同，则记录本配置对本次随机状态变化的敏感性。过程对照无法单独归因于 GPU kernel、区分具体 dropout site，或证明 `.1171` 的历史差异来源。B 单个替换状态未命中也无法排除其它 dropout 样本的影响。该诊断不产生优化器更新、checkpoint、新训练资格或性能主张。

## 运行接线

- 后续应由新的一次性 controller 在同一合格八卡 Slurm step 内按 `a1,b,a2` 串行启动三个新的原生 Swift/torchrun 进程组。每臂 fresh output 到 `<new_root>/<arm>/train`。三组前后都保完整强 PID / startticks、真实 PCI 各 rank 映射与清场，未清场即保失败。原 `.1189` scope 保持封存。
- 每臂只替换诊断 external plugin 为 `native_rng_observer.py`、设置单独 output，并提供该臂 `MINICPM_RNG_TREATMENT`。保留原 witness 名 `minicpm_usgs_resume_witness`、原冻结父 witness 和原 `MINICPM_USGS_*`；新环境变量 `MINICPM_FIRST_FORWARD_ORIGINAL_OBSERVER` 指向已审查、只读的原 `native_observer.py` 与 `probe.py` 所在目录。 `rng_treatment.py` 与新插件同目录。三臂复用字节相同的 checkpoint witness 配置。先统一锁定设备拓扑和单一预算。
- 原三类 receipt `first-forward-state/ready/stop.json` 与新增 `rng-treatment.json`、`rng-state-audit.json` 必须齐全。聚合在消费真实 Slurm 终态后运行，另做独立 post 驗证再释放节点。

```bash
/usr/local/python3.12/bin/python3 -I -S -B compare_bracket.py \
  --root "$NEW_ROOT" --gpu-step "$ACTUAL_GPU_STEP" \
  --original-route "$PINNED_1189_ROUTE" --original-root "$ARCHIVED_1171_ROOT" \
  --original-public "$PINNED_1171_TERMINAL" --original-observer "$ORIGINAL_OBSERVER"
```

本包为本机源码与验收逻辑。后续 HOST 接线、现场准入、设备执行与性能结果仍待验证。输入 fixture 和 generator fixture 仅在本机临时目录运行。
