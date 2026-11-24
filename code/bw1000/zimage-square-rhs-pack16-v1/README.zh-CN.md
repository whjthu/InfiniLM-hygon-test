# Z-Image 同一 RHS 的布局复制与 16 次复用

原训练 `.1787` 将实际 `(M,N,K)=(288,3840,3840)` BF16 `aten::mm` 的两种 RHS stride 分别关联到 120 次调用。LHS 均为 `(3840,1)`，RHS `(1,3840)` 对应原 `Cijk_Alik`，RHS `(3840,1)` 对应原 `Cijk_Ailk`。原 `.1877` 的两次独立 stock trace 分别观察到 104.960 和 77.120 µs。它们用于选择实验，尚未测得同一 RHS 的复制成本或可比吞吐。原 `.1787` 采集器保存 shape、stride、dtype、device、layout，缺少持续 RHS 身份、版本或变更顺序，真实训练中能否复用 16 次仍未知。

两个试验臂都用 stock `cublas` 选择和同一逻辑 RHS。A 直接读原转置视图。B 在同一 stream 一次 `copy_` 到预分配的连续视图，然后用它做相同 16 次 `torch.mm`。两份受控合成 LHS 交替，完整输出逐位比对独立 CPU FP32 参考后转 BF16。先用独立 hipprof 证实两个真实调用各出现一条匹配原训练的 Cijk 路径，再从新无 profiler 进程状态按 AB、BA、AB 完成三对。每臂 2 个预热 burst、4 个计量 burst。主指标是含复制及同步的 burst 宿主用时，GPU event 分别记录完整 burst、复制区间和 GEMM 区间。全部计量调用共 384 次，packed 臂 12 次计量复制。输入和 packed 存储检查 guard 与逻辑内容，普通输出检查完整结果。每臂单列实际 Torch peak allocated/reserved。

packed 持久载荷额外 29,491,200 字节，两端 guard 再增加 1,024 字节。每次复制的逻辑读写合计为 58,982,400 字节，实际 HBM 流量待量测。试验中两个臂都保留预分配 packed 缓冲，实际生产选择的显存增量单独评估。三对完整且含复制宿主速度比中位数至少 1.05、每对至少 1.0 时，才继续复用位置验证。即便组件过门槛，整模型收益和训练效果均需另证。

本机 `python -I -S -B -X utf8 check_local.py -v` 验证渲染、实际 caller 顺序、别名拒绝和全部分母。`layout_contract.cpp` 的 Windows g++8.3 编译和执行均返回 0，检查 14,745,600 个 BF16 存储元素、边界保护和 9 个抽样点积。原本机回执 SHA `20dcdedc233725dcd3bb61deff0dd8695272937f866ab68d87bf6a9f705cd750`。本轮实际 target 编译、GPU、profile 和 Slurm 均为 0。

采用方先 ordinary published claim，运行 `prepare.py --out <新目录>`，单独验证目标 DTK/hipprof/PyTorch 来源和 runtime manifest，再以 `worker.py --scope <原件> --scope-sha256 <实际 SHA> --out <新私有目录>` 启动。运行包不提供未授权调度器入口。需另有新鲜独立前后清场、实际 1 DCU 租约、单 epoch 和生产终态。`f12r2n11` 由原 `.2559` 保底保护，当前包主动拒绝该节点。原 `.2175` OOM、`.2292` 权限失败及 `.2299/.2424` 零卡实测保持封存。
