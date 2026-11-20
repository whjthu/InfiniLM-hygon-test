# GLM I128 编译收据核验

两项固定 CPU 编译资源独立。gate/up 从 4 行改为 1/2 行，节点为 `1103606/f13r1n09`。down 从 8 行改为 2/4 行，节点为 `1103606/f13r1n08`。每项为 4 CPU、8 GiB、0 GRES、300 秒，保留各自的唯一 label 和两个不同编译参数 ID。各自的 gate/up [形状和回退](../routed-i128-gate-rows-CPU-v1/README.zh-CN.md) 和 down [形状和回退](../routed-i128-down-rows-CPU-v1/README.zh-CN.md) 仍按原认领执行。

从仓库运行 `python -B -X utf8 <本目录>/audit_once.py inventory`，验证两组 manifest、任务身份、源文件和四个参数 ID。输出不做实时准入，也不提交作业。认证恢复后，协调者走各原包的 `launch.py.main(publication, transport)`。先按原请求、原 label 和各自节点的实时状态去重。

审计一次已知终态时，在私有目录保留原 launch JSON、Slurm accounting 和构建结果树。Accounting 输出字段固定为 `JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES`。可使用原 launch 的集群路径或不改变目录结构和文件字节的本地封存件。

```text
python -B -X utf8 <本目录>/audit_once.py audit down --launch ORIGINAL-LAUNCH --accounting ORIGINAL-SACCT --output-dir ORIGINAL-CANDIDATE-DIR --publication EXACT-PUBLICATION-COMMIT
```

成功要求原 Slurm 步骤退出 0、申请资源一致、两臂的原编译命令和日志完整、所有 fresh M 先于编译、M 与实际 MD 路径集合相等、对象及源码实读 SHA 一致、三项 C ABI 和主机 ELF 对象类型匹配。报告未压缩 AMDGPU 内嵌 ELF 的原始 flags 和 SHA，不能据此独立认证 gfx936 ISA。报告完整保留失败或缺失构建证据的两臂分母。启动目标编译、读取私有原权重、实际设备数值和模型推广均由原独立门槛决定。

本地 `check_local.py` 用 g++ 生成真实 x86-64 主机对象，再用本机 objcopy 转为 ELF。测试向收据读取器注入明确的模拟 Slurm 和编译收据，仅用于检查通过和拒绝路径。不会从本机 ELF 推导集群目标编译、设备执行或 GPU 数值。
