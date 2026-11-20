# GLM TP16 routed I128 down 的 CTA 行数

原 GLM 5.3 TP16 完整步骤 `1103606.310` 的深度观察覆盖六个稀疏层，每层 16 个 rank。routed down stream 包围窗中位 55.53–56.32 µs，单 rank 最大 100.33 µs。下一步用同流设备事件测单叶，不将历史窗口当作候选速度。

仅改变同一 CTA 内独立输出行的数量。原 8 行 / 256 线程 / 768 个 X blocks 复用 CPU1087 的实际对象。新 2 行 / 64 线程 / 3072 个 X blocks 和 4 行 / 128 线程 / 1536 个 X blocks 都使用原 W32/V4 树及每行独立启动的缩放、转换和输出。每两个 32-lane subgroup 保留原分组语义。原 gate/up、validate、FP8 解码顺序、F32 树和 BF16 转换均相同。与旧 TP8 down row2 复用路线不同，此处不共享多个输出行的激活解码，本轮也不融合上游 quant 或修改 CUDA stream。

| 实际张量 | 形状 | 元素 stride | dtype |
|---|---|---|---|
| route activation | [8T,128] | [128,1] | E4M3 原始 UInt8 |
| route scale | [8T,1] | [1,1] | FP32 |
| route IDs | [8T] | [1] | I64 |
| expert down weights | [256,6144,128] | [786432,128,1] | E4M3 原始 UInt8 |
| expert down scales | [256,48,16] | [768,16,1] | FP32 |
| down output | [8T,6144] | [6144,1] | BF16 |

两个候选在同一个 HOST CPU 步骤编译。通过原 CPU1087 的 DTK 路径核验，`nvcc -std=c++17 -O3 -arch=gfx936 -Xcompiler=-fPIC`，所有新候选 fresh M 都与生产者图相同后才开始正式编译。逐候选核对实际 MD、三项 C ABI 和对象 SHA。源、工具链、编译前缀与原对象身份组成去重 ID。保留此前的 gate/up 候选、编译目录、实际模型权重与库。

精确发布后经原认证连接调用 `launch.py` 的 `main(publication, transport)`，回调接受 `(body, stdout, stderr)` 并将 body 送入 HOST 的 `/usr/local/python3.12/bin/python3 -I -S -B -`。采用 `1103606/f13r1n08`，4 CPU / 8 GiB / 0 GRES / 300 秒。身份与节点匹配、无活跃步骤、两父账本确认标签和 namespace 不占用后才提交。候选结果为各自的 `routed_i128.o` 和一个 `CPU-result.json`，用独立 Slurm 终态绑定。此步骤无 SDK 调用或 GPU 数值。

GPU 须新立认领。先比较原 T1 真实 8 routes 及对应 TP16 rank 权重切片，覆盖专家 0/255 中间例、不同 rank offset（0/128/1920）和 T8 回归。检查整个输出与 canary、非有限、原 provider 和 PCI。固定同卡同流，至少三组完整配对，每组 20 预热与 100 次计时，收集事件时间、host submit、峰值与分母。只有正确性通过、三对同向且中位节省 ≥5% 时才进行实际调用方验证。
