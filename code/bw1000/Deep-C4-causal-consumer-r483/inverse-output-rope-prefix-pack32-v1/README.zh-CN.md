# DeepSeek-v4 inverse RoPE 的 BF16 前缀搬运宽度

固定 H1 的每头一个 CTA、64 线程、主负载 8 CTA，沿用 TP8 单 token 的 BF16 `[1,8,512]` 与 F32 `[1,64]` 角表。未旋转前缀每头 448 个 BF16，占输出字节的 87.5%。候选在输入和输出均 4 字节对齐时按相邻两个 BF16 做 32 位位拷贝，仍只读写相同的字节。任一指针仅 2 字节对齐时沿用原 H1 的 16 位搬运，使用同一 stream 和 launch 几何。原尾部逆旋转的 FP32 运算、BF16 舍入、段间重叠拒绝和 C ABI 参数布局保持原样。

Deep r467 两个实际窗口的 inverse-output-RoPE stream 包围分别是 8.084954 和 8.224958 ms。该信号只定位候选，不能由前缀字节比例推导执行时间。Win64 GCC 8.3 对同源码的单对搬运 helper 确实生成了两次 16 位读取和两次 16 位写入，32 位 helper 则生成一次 32 位读取和一次 32 位写入。对应的 gfx936 指令、端到端同步时延和整模型收益仍须实测。

运行本机验证和确定性渲染：

```text
python -I -S -B check_local.py
python -I -S -B render.py <new-target-directory>
```

渲染器检查冻结 H1/H2 原包和真实 profile 的 SHA，仅创建新的 gfx936 单 TU 包。原 target worker 从冻结模板衍生，保留 CPU1113 工具、fresh M 和实际 MD 校验。唯一新导出是 `deep_inverse_rope_h1_pack32_v1`，保留旧 H1 作为后续独立对照。实际目标运行需另由协调者绑定真实空闲资源和新认领，计划为 4 CPU、8192 MiB、0 GRES、300 秒。原 H1/H2 请求与资源独立保存。

本机可执行检查覆盖 18 个对齐用例、3 个对齐回退用例、11 个无 launch 拒绝，以及相同输入、完整输出、红区、同 stream 和两个状态码传播。验证用固定的 CPU 翻译和独立旧参考，未调用目标 SDK 或设备。实际编译后先核 gfx936 产物和对齐分支产生的指令，再做同输入完整 BF16 检验。
