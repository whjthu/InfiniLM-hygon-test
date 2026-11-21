# DeepSeek-v4 inverse RoPE 编译产物独立验收

此入口只读取已存在的原任务结果。任务范围绑定原请求 ed7f8bd81af545252986bd3d25810f73、parent 1103606、节点 f12r2n12、4 CPU、8 GiB、0 GRES 和 300 秒。两组候选为同一个对象中的 H1、H2，主形状为 TP8 单 token 的 [1,8,512] BF16。绑定实际目标成功和失败均保留两个候选的分母。

    python -I -S -B audit_once.py --intent <original-intent.json> --controller <original-accounting-result.json> --accounting <independent-sacct.txt> --build-dir <original-candidate-dir> --producer <original-CPU1113-result.json>

独立 sacct 文件使用 JobIDRaw、JobName%100、State、ExitCode、ElapsedRaw、NodeList、AllocTRES 七列，以管道符分隔。controller 如曾记载原作业终态，须与独立终态一致。controller 保留 UNKNOWN 时，随后可凭实际 Slurm 终态、产物及 worker 回执消费原作业。失败终态不读编译或 GPU 产物。命令本身无 Slurm、SSH 或启动逻辑。

成功路径核对原 DTK gfx936 预处理、编译和符号扫描命令及 stdout/stderr 哈希，解析两个原 Make 依赖文件的对象和路径集合，使用 CPU1113 的原依赖清单逐项查 pin，随后检查原对象的 x86-64 REL、强定义函数表和两组原 nm 结果。实例若含可解析的未压缩 AMDGPU ELF，报告其实际大小、SHA 和 flags。目标架构实证、链接、加载和数值仍需后续实际设备验证。

本机测试的 host 机器码由 g++ 生成，再由 objcopy 转换为 ELF。因 COFF 不保存 ELF 符号大小，测试按真实函数偏移为 fixture 写入符号大小。测试覆盖自洽假对象、缺入口、依赖漂移、错架构、重复 accounting、非终态、既有 controller 冲突，以及在最初 UNKNOWN 后消费同一原件。Slurm 和旧 producer 的测试记录均为本机 fixtures。
