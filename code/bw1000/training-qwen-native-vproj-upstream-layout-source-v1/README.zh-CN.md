# Qwen T01 原生 V 投影上游梯度布局诊断

原实机 `.348` 的 BF16 M186 输入梯度中位事件时间为 0.23664 ms，M192 padding 为 0.40524 ms。同期 600 卡合成 dX 也已排除 padding。原 `.348` 的 `vproj_input_vjp.py` 在 `keep_upstream` 中先 clone，再由 `replay_state.describe` 保存 shape/dtype/SHA，历史证据缺实际 SDPA/PEFT 下游传入 dY 的 stride、offset 和存储边界。O01 depth9/15 的固定地址及重复逻辑对无法补足这项原调用证据。

下一项新且已获准的原生 T01 微批次观察 rank 1 `decoder_06.v_proj` 的 frozen base linear。取实际 BF16 `X=[1,186,4096]`、冻结无 bias 的 `W=[1024,4096]`、真实 base 输出和实际 autograd 上游 `dY=[1,186,1024]`。仅当新原生 forward 与 backward 都命中该入口才接受数据。原 hook 保留 dY 的暂时私有引用，记录正 stride、offset、存储范围、指针对齐、版本及 owner stream。源布局重叠、范围超过 8 MiB、状态变动或实际输出版本变动时，停止诊断。原 LoRA、SDPA 和训练路径沿用当前配方。

同一次自然 backward 之后，用相同 `torch.nn.functional.linear` 和 `torch.autograd.grad` 对同一实际叶子回放。原 dY 已连续时只记原 forward/dX 阶段并关闭布局轴。如果确证正 stride 非连续，使用 12 个计时样本，按原视图、连续 pack、连续 pack、原视图的顺序，每臂先执行一次预热，再计三次。B 的 380,928 字节静态 scratch 预分配，复制发生在 `pack_plus_dx` 计时区间内。全量投影输出及 dX 必须与真实 native edge 逐位相同，packed 样本还核对 dY 全量逐位相同。报告事件 forward、pack、dX、pack+dX，及包含等待的 host wall。事件区间包含同一流提交等待，无法归属 GPU 独占计算时间。A/B 均调用相同四个事件且使用相同 GPU 及固定 M186。若两个配对的 pack+dX 时间都至少减少 5%，才保留为局部候选，实际 host wall 与一次新原生训练验证仍必需。任何前缀失败保留分母。

## 现有调用接线

使用当前冻结 `training-qwen-native-vproj-native-row-vjp-v1/vproj_input_vjp.py` 的 `ProjectionInputGradient`、其 `text_base_preference.py` 的实际 selected witness hook 和原 `torch.nn.functional.linear`。新的已准入 caller 只将创建 rank 1 recorder 的位置换为 `native_observer(ProjectionInputGradient, torch, linear)()`。继承的 `capture_on_view` 保留真实 autograd 叶子、输入和 dX hook，新增 hook 返回 `None`。只在原 backward 退出后调用 `evaluate(original_absolute_deadline)`，清理原诊断张量并保存小型 JSON。未命中真实 shape 时留独立不合格记录，无旧输入重跑或填充。旧 `.348` loader、`checked_archive` 和原启动冻结。

该目录没有 target launch、GPU admission 或 target compile。用固定本机 Python 运行 `-I -S -B -X utf8 check_local.py -v`。验证使用边界 fixture，以及实际计时函数的模拟设备调用顺序。训练设备结果与候选收益仍待新真实数据。
