# GLM TP16 I128 一卡组件验证

入口使用原 r470 叶子 ABI。对照件为 ORIG，G1/G2 只对比 gate/up，D2/D4 只对比 down。五份隔离的目标 DSO 均应来自相应的真实 gfx936 对象，并由外层独立审计绑定。这里保留与实际调用一致的三个 C ABI，逐个句柄解析并通过 `dladdr` 核对归属。HIP 函数同样核对持有人。

## 输入与数值

| 缓冲区 | Dtype | Contiguous shape |
|---|---|---|
| route IDs | I64 | [1,8] |
| gate 输入、输入 scale | E4M3 packed U8、FP32 | [1,6144]、[1,48] |
| gate 和 up 权重、权重 scale | E4M3 packed U8、FP32 | 各 [256,128,6144]、[256,16,48] |
| gate 和 up 输出 | BF16 | 各 [8,128] |
| down 输入、输入 scale | E4M3 packed U8、FP32 | [8,128]、[8,1] |
| down 权重、权重 scale | E4M3 packed U8、FP32 | [256,6144,128]、[256,48,16] |
| down 输出 | BF16 | [8,6144] |

四组固定输入对应分片 offset 0、128、1024、1920。路由各有八个不同专家，涵盖紧凑、稀疏分散、重排和专家编号两端。三个完整权重组按实际 ABI 物化，未选中的 248 个专家的权重载荷为零。选中的行按固定字节模式和非均匀 FP32 scale 生成。每工况占 606,346,528 B 合成输入。只保留输入 SHA 和输出统计，down 输入独立生成，范围为两个叶子。输出到完整模型、训练数据或服务质量的外推待实际整模型实证。

原 ABI 使用 unsigned byte 作 E4M3 载体。独立 CPU oracle 从有限 E4M3 与 dyadic scale 算精确整数点积，核对所有 204,800 个基础参考 BF16 值。叶子数值门禁要求有限、数值差异零，正负零位型差异单列。四组候选各自还与对应 ORIG 输出核对。验证入口用固定 I64 ID 做正常验证，并分别送入 -1 和 256，非法 ID 只走 validate，结果经拥有者流同步读取。每个输入输出缓冲区有首尾各 256 B 的哨兵，输入完整 payload 在每工况末从设备回读核 SHA。

## 执行与计时

并行关系为 TP16 的指定本地分片 ABI、T1、8 路、单物理卡和同一个 owning HIP nonblocking stream。用 `hipGetDeviceCount` 与 PCI 位置核对外层准入，支持 HIP 省略零 domain 的 PCI 表达。分配前、驻留和释放后检查采样空余至少 8 GiB。主机合成、输入分配、H2D、预热、验证、D2H 及清理单列在计时之外。

每工况每候选以 ORIG/候选、候选/ORIG、ORIG/候选做三个完整配对。每臂预热 20 次，重新清空 BF16 输出，接着用同一流上的真实 HIP event 包住 40 次已提交调用。另存从首个 event 提交至尾 event 同步返回的主机耗时。event 区间可能包含该流等待主机提交的间隙。结束后独立重查 BF16 与哨兵。每候选每工况要有三个完整配对和全部输入回读证据才记录配对中位数。保留 24 项数值门禁、48 对、96 臂和计划 3,840 次计时调用的原分母。

```text
python -I -S -B caller.py --binding /absolute/path/to/admitted.json --out /absolute/path/to/new-private-result.json
```

`--binding` 使用 `schema: "glm_i128_one_gpu_v1"`、`hip: {path,sha256}`、`providers: {ORIG,G1,G2,D2,D4}`、`expected_pci`、`epoch`、`deadline`。每个 provider 条目含绝对 canonical DSO 路径和 SHA256。任意件缺失、为空、非 canonical、SHA 不符或身份重复时，在加载任意依赖或接触 GPU 前结束。预注册分母先写入独立结果，完成或阶段失败后原子更新。外层负责已验证目标产物、唯一作业、Slurm 硬截止、进程终态以及独立清理。caller 用同一 launcher deadline 和本地 monotonic 补充检查，单次阻塞的驱动调用由外层时限限制。

`check_local.py` 真实物化合成输入并检查注册 SHA，以独立浮点解释验证 36 个标量输出，另测本机哨兵、绑定拒绝、PCI 归一化、计时事件顺序和失败分母。测试使用本机记录器模拟 HIP 调用。
