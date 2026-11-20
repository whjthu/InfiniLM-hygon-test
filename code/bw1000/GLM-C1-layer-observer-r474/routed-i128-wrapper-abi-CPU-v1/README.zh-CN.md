# GLM TP16 I128 主机 ABI 验证

使用实际 `routed_i128.hpp`、CPU1087 原 CUDA 翻译单元及 gate/up 1、2 行和 down 2、4 行四个冻结候选。读取 .310 完整 TP16 模型的原深度 profile 及调用源码，核对每个叶的六层 × 16 rank 样本。对应 stream 包围时间分别为 gate/up 173.84–176.48 µs 和 down 55.53–56.32 µs。

本机验证只转换三处 CUDA triple-chevron 启动为强类型记录调用，再用 g++ 编译整个翻译单元。保留原声明、三个 C ABI、所有 wrapper 分支、参数以及内核源码。分别编译 HIP 与 CUDA 两种 shuffle 预处理分支。记录两个 projection wrapper 的指针顺序、routes、rank offset、同一 stream、零动态共享内存、grid 和 block，以及透传 `cudaGetLastError` 的返回值。非法 routes/offset 在任何启动之前拒绝。原 `validate` 内核由主机循环逐线程执行，覆盖第 1、255、256、257 个 route，以及专家 ID −1、0、255、256。

验证不执行 gate/up 和 down 的 GPU 数学。FP8 解码与 wave intrinsics 只通过本机编译类型检查。逻辑 lane shuffle 与 BF16 由测试替身提供。输出中的 HIP 与 CUDA 指代两个预处理分支。

从本目录或仓库运行 `python -B -X utf8 check_local.py --cxx g++`。两组独立的目标认领、形状、精度、参数和回退仍在 [gate/up](../routed-i128-gate-rows-CPU-v1/README.zh-CN.md) 与 [down](../routed-i128-down-rows-CPU-v1/README.zh-CN.md) 原包。恢复原连接后先核对原请求、Slurm label 与当前节点，然后使用各自的原编译入口。消费真实目标编译终态时用 [终态审计工具](../routed-i128-cpu-artifact-audit-v1/README.zh-CN.md)。之后另行做实际 checkpoint 数值配对与性能验证。
