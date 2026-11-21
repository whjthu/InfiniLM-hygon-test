# GLM-5.3 TP16 新到 K/V 合并写入

原 .310 整模型真 AR T1 的六个已采样层，fresh.expand_cached 流区间中位数为 0.196–0.221 ms。CPU308 构建的准确源码 u15/u17 显示，该区间先做 kv_b 全宽 FP8 投影，再把结果切成非旋转 K 及 V，并把共享旋转 K 分别复制到四个本地 head。区间包括投影和原同步，实际拷贝成本尚未拆分。

本候选仅在投影后组装新到 K/V。实际 TP16 量级为 T1、4 个本地 head。输入 BF16 contiguous [T,4,448]，元素步幅 [1792,448,1]。单份共享旋转 BF16 输入 [T,1,64]，步幅 [64,64,1]。输出 K、V 各 [T,4,256]，步幅 [1024,256,1]。K 的前192列来自投影，后64列来自当前 token 的共享旋转 K。V 对应投影第192到447列。各 buffer 独立且已由调用方按真实 dtype、设备和 stride 校验。

两个 C ABI 是 glm_r474_fresh_kv_pack_t64 和 glm_r474_fresh_kv_pack_t128。唯一变量为每 CTA 的 BF16 对数，T1 对应 8 和 4 个 CTA。对当前 T1，两组均把 1 次非旋转 K 复制、4 次旋转 K 复制和 1 次 V contiguous 组装合并为一次提交。实际设备的数据流和 launch 收益待测。原 FP8 投影及其完整1792输出、历史 selected-V、Runtime 流及同步须在未来接入时沿用。候选只准备单独对象，生产路径仍由原库提供。

本机把 CUDA 启动替换成记录器，编译运行实际复制主体，按原三个切片动作独立核每个 BF16 位。接受范围 T1–512，head 固定为4。T2/17/512 仅作为组件控制。测试覆盖多 token 的 rotary 行定位、输出红区、不可变输入、空流、错形状、交叠范围、非对齐和原 launch 错误传播。

    python -I -S -B prepare_target.py <new-target-source-dir>

目标源码包包含核、固定 PLAN 和由已审编译脚本按精确字符串替换生成的 compile_once.py。目标只建一个 gfx936 对象和两个 C ABI；采用 CPU1113 实测的 DTK 工具与依赖记录，再核此次 fresh M、actual MD 及实际强符号。由根协调者按 4 CPU、8192 MiB、0 GRES、300 秒选取可用 CPU 节点，传既有编译入口需要的 TUNING_CPU_PARENT、TUNING_CPU_NODE、TUNING_CPU_MANIFEST_SHA、TUNING_CPU_EPOCH 和 TUNING_CPU_DEADLINE。提交及硬截止仍由根控制。首目标对象的路径为 /refactor/candidates/GLM474-fresh-kv-pack-gfx936-CPU-v1/fresh_kv_pack.o。

后续设备比较需要原 checkpoint 和同一 TP16 调用、原完整 FP8 投影结果、真实 Runtime 流及同一上下文。两组都先校验全部输出位、输入和显存哨兵，再比较至少三个完整配对的组装区间及完整 fresh.expand_cached 墙钟。拷贝降时可以单独记录，完整区间与模型收益由相应测量决定。
