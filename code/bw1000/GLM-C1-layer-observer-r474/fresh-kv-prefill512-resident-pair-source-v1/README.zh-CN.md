# GLM r474 全权重一次加载的首段 T512 对照

范围是固定 TP16、PP1、C1、32K 模型入口，绑定实际零 GRES Native 构建 `1103606.1071`。每个 owner 通过原 checkpoint loader 完成 117060 项、78 层加载，再在同一个 context 中按 OFF、ON 两次消费原 ordinal 1 的完整 1312 个 ID。两个臂均为 512、512、288，使用同一段 resident 输入、位置、输出和真实非默认 producer stream。仅 ON 的首段调用 `modern_glm_global_fresh_kv_prefill512_r474`。原始 ID、参数和 logits 保留在目标集群，报告只写摘要。

每个 owner 记录首段 CABI、三段 CABI 总和及带观测开销的完整宿主循环耗时。首段三类回退连同成功数保全 78 层分母，跨 16 个 owner 共 1248 次。rank0 同时保留 OFF 三个完整的 154880 元素 BF16 logits，逐字节比较同位置的 ON 输出，并验所有元素有限。最终验证还要求每卡完成两条完整 prompt、三次 reset、关闭原 context、8 GiB 边界样本及持续 driver 样本。单序 OFF、ON 允许验证实际入口与数值，速度差须另做多组配对。此范围停在 prefill，无输出生成、HTTP 和正式效果验收。

## 目标接线

- 由新的受限 HOST/controller 提供两节点同组 TP16 资源、16 个 fresh PCI、原清理边界和单一绝对截止。旧 .906 节点、端口、预约及结果路径属于已结束的原作业。原 keeper 的独立 Slurm 终态与物理清场仍需接入。
- 每个节点在相同挂载与已固定环境内使用现有有界 ELF 依赖读取，再调用 `bindings.verify_node_provider_bytes` 验证实际 recipe 的全部 71 条路径与 SHA。每个 owner 在正确物理设备和主线程上调用一次 `bindings.load_native`，校验实际 .1071 ELF、生成的 typed 字节及 14 个 CABI 的活动提供者。
- rank0 通过 `bindings.unique_ids` 取得原 CABI 产生的两个不同 UID，既有协调层以本次身份一致传递至所有 owner。随后 `run_pair.run_owner` 接受已经绑定的 Native、UID、原 checkpoint loader、原输入绑定、fresh PCI、截止、同步栅栏、容量/持续显存采样及写盘回调。`memory(label)` 返回原 driver 的 `(free,total)`，`memory_summary()` 接原 5ms sampler 的最终汇总。设备上的输入 ID 和位置在首臂前及每臂后逐值核验。正常关停经过 `prefill_pair_closed` 栅栏，异常由 keeper 收回所属进程。
- controller 以 16 个正式 owner 的最终报告及其 fresh PCI 构造 `aggregate.summarize(records, admitted_pci)`，保存原记录与本次 GPU、keeper、清场证据。单个 Python 源包没有集群准入权限，`SCOPE.json` 仅固定本地范围。当前无 GPU 派发。

## 本机验证

`python -I -S -B -X utf8 check_local.py` 使用本地真实 .1071 公开构建证据、原输入元数据与重渲染 typed 字节。六项回归用 CPU 录制对象检查实际调用安排、单 context、失败分母、排布、显存、完整 BF16 行及 16-owner 汇总，目标 Native/SDK/模型/设备调用为零。
