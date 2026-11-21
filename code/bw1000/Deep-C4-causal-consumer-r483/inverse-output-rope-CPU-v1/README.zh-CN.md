# DeepSeek-v4 attention inverse RoPE 融合

DeepSeek-v4 attention 输出路径先复制整个 BF16 张量，再用 InfiniOps 在副本上做 inverse RoPE。原 C1 .1195 实测该包围为 6.91–7.30 ms，后续 .1476 归档的两个窗口为 8.08 和 8.22 ms。区间含 Runtime 边界、拷贝、查表准备、调度与同步。当前优化目标是删除其中一次数据复制及尾部重复数据读写。

两个独立 C ABI 为 deep_inverse_rope_h1_v1 和 deep_inverse_rope_h2_v1。签名与参数次序为 int(void *out, const void *input_bf16, const void *table_f32, size_t tokens, size_t heads, void *owning_stream)。主形状为 TP8 AR [1,8,512]，BF16 元素步幅为 [4096,512,1]。另测 T17/128/512 的同布局组件控制及 TP1 T1/17。支持 T1–512、head 数8或64，之后的集成调用方需保留原路径覆盖其它尺寸。

输入张量与目标均为连续独立分配，已完成原 position 验证和 FP32 查表上载。输入 table 为连续 [T,64]，每行前32个 cos、后32个 sin，逻辑表行就是本 token 的行号。按原 provider 的非 NeoX 语义做相邻偶奇道的旋转，inverse 保持先取负 sin，结果写回 BF16。前448值逐位复制。调用方保留原 host position/YaRN/表、流拥有及异常清理。

H1 每 CTA 含1个 wave64 head，H2 含2个 wave64 head。每 head 的逻辑访存及计算一致。T1/TP8 网格分别为8和4 CTA。合并后每次调用少一次先行 copy 调用和逻辑重复的 2 KiB 尾部读取加写入。T512/TP8 对应 1 MiB。按同一个输入、设备和原 SDK provider 的配对测试判断实际收益。

本机检查替换 CUDA 启动和位转换接口，实际编译执行完整候选 C++ 主体。FP32 本机禁用乘加融合，用于验证数组索引、相邻道、逆旋转符号和舍入。本机查表为合成固定表。目标 SDK 的编译收缩方式及真实输入基线待设备验证，首轮在同一卡和 stream 上保持原 provider 与两臂成对运行，保留原表和 I64 位置，检查输入及输出哨兵、前448逐位、尾64有限及 BF16 误差分布、失败分母。再看原 owner 区间中位数和三轮方向。

## 有界编译入口

在目标 DTK 容器现有 CPU-only Slurm 步中调用下面的入口。

    /usr/bin/python3 -I -S -B -u /refactor/incoming/Deep-inverse-output-rope-gfx936-CPU-v1/compile_once.py

根在 HOST 确认唯一命名空间、有效分配及正式提交，申请4 CPU、8192 MiB、0 GRES、300 秒。进入容器后沿用已验证 DTK 环境并禁用设备枚举，传已准入的 TUNING_CPU_PARENT、TUNING_CPU_NODE、TUNING_CPU_MANIFEST_SHA、TUNING_CPU_EPOCH、TUNING_CPU_DEADLINE。epoch 采用一次真实启动时间，原期限为 epoch+300，外层 timeout 和 Slurm 硬期限需同源。父项、目标节点和原作业步由真实外层证据填写。

编译前重验原 CPU1113 工具链、原19份依赖闭包和候选包 SHA。只有实际 fresh M 中的每个路径及文件 SHA 均已知才编译，之后比较实际 MD 并检查导出符号。目标生成一个对象 /refactor/candidates/Deep-inverse-output-rope-gfx936-CPU-v1/inverse_output_rope.o，同目录的 CPU-result.json 保存命令、SHA 和资格。本机完成记录见 LOCAL_RESULT.json。
