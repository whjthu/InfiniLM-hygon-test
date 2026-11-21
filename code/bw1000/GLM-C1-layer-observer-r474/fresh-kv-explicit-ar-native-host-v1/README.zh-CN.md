# GLM 显式自回归三编译单元的 HOST 入口

这个入口绑定原请求 `eca8f3dcb6d3d3938ce50868dfe670fd` 和已发布的三编译单元 worker `ec92243f2c9fa4573f8029197448334ef396a84e`。在 `1103606` 的 O01 节点 `f12r2n13` 使用 4 CPU、16 GiB 和零 GRES。一次编译 `u00`、`u15`、`u17`，复用 CPU308 的 29 个对象与原 gfx936 对象，链接闭包含 33 个对象。

提交前核对同一 pack 请求 `b9a7f0b78c566911ed40004fc906ce44` 的真实记账、输出和设备位置。旧步骤须从两项父作业的活跃步骤中消失。HOST 还复查当次节点归属、父作业剩余时限和本请求的实际 Slurm 标签。根协调者在项目调度账本登记原请求后，才从已核验的目标项目 checkout 调用本入口。

```text
/usr/local/python3.12/bin/python3 -I -S -B <checkout>/code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-explicit-ar-native-host-v1/host_once.py
```

原意图先独占落盘，worker 和 Slurm 使用同一 epoch，总预算 1800 秒，编译工作至第 1680 秒。目标目录、Slurm 标签或本地原意图已存在时，保留原请求并走 `--reconcile`。HOST 结果与独立 ABI 检查分开，实际编译终态再由 `tools/read_glm474_fresh_kv_explicit_ar_native_terminal.py` 消费。首次目标编译和 GPU 执行尚未发生。
