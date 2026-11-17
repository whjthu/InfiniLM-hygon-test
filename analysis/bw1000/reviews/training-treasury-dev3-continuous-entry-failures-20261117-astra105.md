# Treasury 连续入口失败复审（Astra105，2026-11-17）

**本次已证事实是 base 原 GPU stderr 第2行 LLVM ENOSPC；具体文件系统/缓存路径、inode/quota 状态及 Model 是否构造仍 UNKNOWN。没有 Python traceback，不能归因为 Swift/API/import 错误。停止 GPUv4、重试及改名重播。**

原请求 `3012fc14dd404fe39ffd12ce838a1154`，pub `99c1e7fda254da6a36042c320bd245f7db4ddfdf`。[根完整独审](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/local-only/training-qwen-treasury-explicit-dev3-gpu-v3-root-audit-v2/public-summary.json>) SHA `9e7fb38f21bd6eb7a2eea92b91a2bf69367765bf2a5c645705e7ed32ec1ab18b`；短日志证据 SHA `574d4f9c355f6569f329097b8d74ddc189393f684467ee846fc894d1dcb4f66f`，原 stderr SHA `ce53c86db295e6276bf7d851c2049f1950259b486d733ce20ef00344ca8ddfa7`。原文：`LLVM ERROR: IO failure on output stream: No space left on device`。仅 base 短日志证明该错误，不能无证推广为另外两 arm 的相同 fatal。

| 原 scope | 实际结果与保留分母 |
|---|---|
| v1 | 3 GPU input/media 入口失败；324 missing |
| v2 | srun stderr 混入 stdout 导致 HOST JSON 失败；0新增GPU；324 missing，旧 UNKNOWN 不改写 |
| v3 | base346/26s、old_domain345/27s、domain172/31s均FAIL；post347/348/173全C0、KFD0；0生成/324 missing |

v3三 arm 均已过39文件/108输入门禁及14份模型文件SHA；这些证明不等于模型构造成功。accuracy保持null，不能称优化轮、有效训练或效果进展。

已确认的**独立资格缺口**：[旧 probe:65](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-qwen-treasury-dev3-v2-input-gate-CPU-v1/probe.py:65>) 用 `run_name="frozen_native_entry_gate"`，只调 `bound_inputs`；[controller:238](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-qwen-treasury-explicit-dev3-gpu-v3/arm_controller.py:238>)同样只验证该子函数。直到 [controller:282](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-qwen-treasury-explicit-dev3-gpu-v3/arm_controller.py:282>) 才设置真实argv并以 `__main__` 执行；此时已过HIP分配及模型SHA。实际 [native:114](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-qwen-treasury-explicit-dev3-gpu-v3/native_generation.py:114>) argparse，以及57–81的versions/ENGINEpins/providerimports/设备准入/输出I/O未被旧输入门禁覆盖。576次新数据encode/collator资格也不覆盖推理main。**该缺口不是本次ENOSPC因果证明。**

**唯一当前候选：三原节点的0GPU exact-container容量与写路径定位。** 复用 [controller:175](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-qwen-treasury-explicit-dev3-gpu-v3/arm_controller.py:175>) 的contain/rocm、binds、PYTHONPATH与环境，分离stdout/stderr；逐节点解析容器内实际temp/cache/output与挂载来源，读取可用bytes/inodes及可读quota，区别宿主与contain视图。只在授权的新exclusive目录做有界写入探针；若需定位实际失败写入，使用已有有界追踪并仅保留路径/操作/errno。不可凭LLVM文字认定`/tmp`，不可删除未知文件、清理共享tmp/cache或直接改API。

退出条件：精确节点/容器路径和原写目标能对应到容量失败或有界复现；否则保持UNKNOWN/HOLD，不放行GPU。任何后续修正仅针对已定位原因，经独立源审，不自动准入下一GPU。后续HOST-main门禁须共享真实main、显式defaultOFF HOST模式，真实经过argparse/provider路径及SHA/模型与adapter参数/108IDs/输出计划，在模型materialize前停止；CPU不得伪装GPU准入成功。该门禁仍不证明native90–103的load后engine/adapter API。

本次只读本地并写本复审；冻结v3未改，未remote/Git/GPU。大capture值reader仍prepared、CPU_GO=false。机器可读原件：[review JSON](<C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/analysis/bw1000/reviews/training-treasury-dev3-continuous-entry-failures-20261117-astra105.json>)。
