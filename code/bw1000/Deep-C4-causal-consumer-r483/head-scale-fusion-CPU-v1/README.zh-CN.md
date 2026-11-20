# DeepSeek C4 indexer head scale 两个编译组合

原 Deep C4 indexer 在 query feature 准备中先将 BF16 head 投影转 FP32，乘原 score_scale，再乘各 head 的 FP32 query scale，交给原 score 和 TopK。原 8499、8500 两个 profile 窗口的完整 query-feature 包围为 10.54–11.48 ms。该窗口还有投影、rope、feature 归一化和状态回读，保持原分项口径。

候选合并三个逐元素提交，按原顺序保留两次 FP32 round-to-nearest 乘法，并只写同形 FP32 输出。以 `Items=1` 为同一融合实现内的控制，`Items=4` 只改变每线程独立处理的元素数。固定 128 线程、同一 stream、不新增 barrier，维持原 dtype、布局和 score_scale 位型 `0x3c3504f3`。原 SDK 三算子全链仍为独立数值及性能基线。新组合不改变 RoPE、量化、缓存、score、TopK、路径状态及发布默认项。

| 形状 | 来源 | 逻辑元素数 | I1 / I4 X blocks | 省下的两块显式 FP32 临时数据 |
|---|---|---:|---:|---:|
| [1,8] | 实际 TP8 T1 | 8 | 1 / 1 | 64 B |
| [512,8] | 同 TP8 布局的 prefill 控制 | 4096 | 32 / 8 | 32 KiB |
| [1,64] | TP1 组件控制 | 64 | 1 / 1 | 512 B |
| [512,64] | 冻结 r483 的 TP1 chunk 对照入口 | 32768 | 256 / 64 | 256 KiB |

head 为 contiguous BF16、query_scale 与输出为 contiguous FP32，元素 stride 均为 `[heads,1]`，score_scale 是原设备 FP32 标量。支持 tokens 1..512、heads 8/64。公开 C ABI 需要不交叠缓冲区和相同设备 owning stream。非支持形状应在未来调用方走原三算子，当前模型未接入。临时字节是源码逻辑上界，GPU allocator 和性能仍待实测。

本机 `check_local.py` 用实际完整候选翻译单元编译并运行两臂。CUDA 启动和两个舍入 intrinsics 使用测试替身。独立原三阶段 CPU 数学参照逐元素要求 FP32 位型相等，包括正负零及小数。目标设备测试另需原 r483 权重范围凭据与真实 causal trace。保留原 SDK 输出、finite、input/output canary、精确 merged FP32、原 score 和 TopK 状态。设备性能只比较同卡同流成对完成的实际 C4 `prepare` 和内层 CUDA event。每组 20 预热和 100 计时重复，先保持三个完整配对及失败分母。中位在所有实际主样本至少节省 5%、三对同向并通过数值才讨论调用方试验。

新有界目标编译认领为 `1103606/f13r2n02`、4 CPU、8 GiB、0 GRES、300 秒。一个 gfx936 翻译单元内含两个可区分 C ABI。复用校验过的 CPU1113 依赖图和 DTK nvcc 配方。fresh M 原始依赖路径与 SHA 全部已知后才编译，校验实际 MD 全等与目标对象双符号。普通发布后，协调者通过原批准连接调用 `launch.py.main(publication, transport)`。回调签名 `(body,stdout,stderr)`，HOST 目标为 `/usr/local/python3.12/bin/python3 -I -S -B -`。写原意图和唯一标签，接收未知时只按旧请求对账。编译输出为 `candidates/DeepC4-head-scale-fusion-gfx936-CPU-v1/CPU-result.json` 与 `head_scale.o`。不调用模型或 GPU。
