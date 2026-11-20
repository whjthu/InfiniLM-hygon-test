# GLM TP16 routed I128 gate/up 的 CTA 行数

原 GLM 5.3 TP16 完整步骤 `1103606.310` 的深度观察覆盖六个稀疏层，每层 16 个 rank。gate/up stream 包围窗中位数为 173.84–176.48 µs。这个窗包含 stream 上的整体间隔，实际新候选的设备时间以单独配对为准。

固定原 CPU1087 已编译的 `routed_i128.cu`、两个输出和 ABI。仅改变每 CTA 并行运行的输出行数。原 4 行、256 线程、32 个 X blocks 直接复用。新 1 行、64 线程、128 个 X blocks 与 2 行、128 线程、64 个 X blocks 各编译一次。原 lane64 两元素、128-elements 固定 FP32 树、48 个 scale block、FP8 解码顺序、BF16 转换、路由校验和 down 都保留。旧 TP8 分块、W32、激活融合属于其他调用路径，此处保持原 TP16 路线。

| Tensor | Shape | 元素 stride | dtype |
|---|---|---|---|
| packed hidden | [T,6144] | [6144,1] | E4M3 原始 UInt8 |
| input scale | [T,48] | [48,1] | FP32 |
| route IDs | [8T] | [1] | I64 |
| gate/up weights 各一 | [256,128,6144] | [786432,6144,1] | E4M3 原始 UInt8 |
| gate/up scales 各一 | [256,16,48] | [768,48,1] | FP32 |
| gate/up outputs 各一 | [8T,128] | [128,1] | BF16 |

HOST 一次派发，`1103606/f13r1n09` 4 CPU / 8 GiB / 0 GRES，300 秒。复用原 CPU1087 的 DTK `nvcc -std=c++17 -O3 -arch=gfx936 -Xcompiler=-fPIC` 命令前缀，校验工具链和原二进制。两臂全部 fresh M 等于原依赖路径和文件 SHA 后才编译。实际 MD 复核对应 fresh M，三项 C ABI 保留。编译输出留独立 namespace。候选 ID 绑定源码、工具链、源对象和编译前缀，单个作业内两个固定编译组合，不能重提执行状态未知的请求。

通过普通发布后，由协调者使用 `launch.py` 的 `main(publication, transport)`。既有连接回调签名为 `transport(body, stdout, stderr)`，将 body 送入 HOST 的 `/usr/local/python3.12/bin/python3 -I -S -B -`。目标 `CPU-result.json` 必须结合原 Slurm accounting 消费。

GPU 数值和性能另立 claim。优先固定实际 token=1、8 routes、16 个 rank offset、连续原 checkpoint 切片与实际捕获的 FP8 输入。完整检查两个 BF16 输出、输入和 canary、非有限拒绝、provider/PCI、短与长输入分开，不继承编译为设备验证。性能比较在同卡同流做至少 3 个完整配对，预热 20、计时 100 次，分别记录设备事件、host submit、空闲/失败、峰值资源。按每臂的最慢 rank 保留结果，只有数值通过且中位节省达到 5%、3 对一致为正才进入调用方评估。
