# DeepSeek CSA 单 HOST 对象增量构建计划

计划从 CPU1113 原始结果提取 TU07 实际编译、链接和依赖清单。重建时把候选源放在新目录，用 `-iquote` 保留旧源目录的引号包含顺序。原 996 条依赖中保留 995 条，更换源并增加一个头文件，预期 997 条。复用其余 49 个原对象，保留原链接顺序、动态库选项和 68 条已记录的传递加载路径。

先用已发布候选的 `render.py` 生成源码和清单。本机可用原 CPU1113 收据和该清单调用 `tu07_plan.py`，只返回小型摘要。需要复查完整命令时，用 `--write-local-plan` 写入新的私有文件。

目标端先在相同容器映射核实父收据、目标源码、实际原 MD、依赖和复用对象。用 `--verify-target-inputs` 执行只读核验。后续单独冻结零 GRES 认领和预算，沿用原 `Native-CPU-v1/TOOLCHAIN.json` 及 `CPU_inside.sh` 的环境前缀。先在该作业内执行计划的 `preprocess`，其 depfile 通过 `--verify-target-inputs --verify-depfile fresh` 后再执行一次 `compile`。真正的 MD 用 `--verify-depfile actual` 复核。链接前复核 49 个对象，链接后比较原强导出、NEEDED 和同环境加载路径及文件 SHA。首个目标预检、正式构建认领和目标执行仍待完成。

本机已用 1,810,762 字节的真实父收据和实际生成的候选源码导出计划。逐参数核实，链接命令只换编译器的绝对入口、TU07 对象及输出库。收据、清单或 depfile 目标被篡改的三个本机反例均拒绝。目标文件、目标 M/MD、Native/设备数值和延迟仍无新观测。
