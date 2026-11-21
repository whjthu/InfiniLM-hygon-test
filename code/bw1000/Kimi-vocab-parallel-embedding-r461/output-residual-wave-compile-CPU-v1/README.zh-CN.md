# Kimi-K3 输出残差评分的 gfx936 编译准备

原 .340 共完成三组同卡配对，实测残差输入为 BF16 [512,9,7168]。原 tile64 方案让各测点空余显存增加 174 MiB，耗时未显示稳定收益。本次比较的是已经实现的两个 mode 2 融合核。两个核均为 4608 blocks、256 threads，只在最后五级采用不同的归约数据交换方式。full-tree 使用共享内存，wave32 使用 shuffle，保留 FP32 加法次序、BF16 归一化中间舍入、norm [7168] 和 projection [1,7168]。源码的整块同步数为 19 和 9，实际时间需同卡测量。

    python -I -S -B prepare.py <new-source-directory>

生成器重验原源码、原证据、两个已有渲染器及全部渲染结果的 SHA。两臂的 component.cpp、residual_method.inc、tiling 和 score_math.hpp 字节相同，wave32 只替换 fused_scores.cu 并增加 reduction.hpp。生成目标目录后保留其中的 SOURCE_FILES.json SHA 用于本次准入。

首次实际编译由根协调者绑定有效父作业和空闲 CPU 节点，使用 4 CPU、8192 MiB、0 GRES、300 秒。容器入口为生成目录里的 compile_once.py。沿用已实证 DTK 环境，传入 TUNING_CPU_PARENT、TUNING_CPU_NODE、TUNING_CPU_MANIFEST_SHA、TUNING_CPU_EPOCH、TUNING_CPU_DEADLINE。deadline 使用实际唯一启动 epoch+300。外层 Slurm 时限和启动用同一预算。原 CPU1113 只提供当时 DTK 工具与可验证依赖的证据，实际新探测仍需要两个 fresh M。两臂全部先闭合，再逐臂编译、核 M/MD、核符号和对象 SHA。

产物目录为 /refactor/candidates/Kimi461-output-residual-wave-score-gfx936-CPU-v1，分别输出 full-tree/fused_scores.o、wave32/fused_scores.o 和同一 CPU-result.json。两个对象都有一个相同的隐藏 C ABI，每个对象之后要分别链接到相同源码的组件。目标 gfx936 编译、DSO 链接和 GPU 执行按各自的实证晋级。

后续设备门禁采用原 checkpoint 的精确 norm/projection 和固定残差，用相同的完整 softmax 与 final GEMM，检查得分、最终残差、输入及设备红区。配对使用同卡同真实 provider，启动及加载单列。原 .340 的 tile64 结果只约束前置形状、硬件及原结论。

本机检验从渲染包提取两个实际 CUDA 翻译单元。Windows G++ 对 ELF hidden 属性发出警告，本机替身只移除该属性并记录 CUDA launch，通过实际 C++ 编译检查几何、流、返回值和非法输入。目标源保留原属性。本机记录器还验证两个预处理先于任何编译，以及两个依赖漂移的失败顺序。
