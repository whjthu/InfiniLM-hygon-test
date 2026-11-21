# T09 既有真实条件缓存的原生训练回归候选

本包只在本机重建、检查已冻结共同框架的 T09.sft 计划，不启动设备或调度。未另造 loop，也未修改原 backend。现有 target 缓存与 Transformer 的历史独立校验支持提出该候选，但不证明目前目录/软件/资源仍合格。

实际调用原 common `build_generation_plan`，核原 launch SHA、393 清单内两个 recipe 的精确字节，并从冻结 framework tar 核 backend 和 checkpoint helper 的 SHA 与同一393清单一致。实际运行的本机检查同时验证缺失 asset binding 被原 compiler 拒绝。没有导入 Torch、读取权重或缓存正文。

原配置 `warmup_steps=5,max_steps=20,validate_recovery=false` 对应 **25 个 optimizer 更新，0 个恢复更新**。此处 warmup 是计时分段，不能擅自称为学习率预热。backend 使用原 Muon，img_attn_q/img_attn_v LoRA r4/alpha8；原 BC/精度/优化器不改。一个缓存文件反复曝光不是25条独立视频；缓存原视频的公开许可证/独立留出来源未由这些收据证明，所以不能当新增公开领域数据或效果实验。

最短后续是 root 决定是否允许独立的缓存训练功能回归，核当前 exact native source/runtime 与 fresh GPU/FD/端口、绑定目标权重/缓存精确身份，再用现有 bounded supervisor 包裹普通入口。**不能原样复用旧 replay controller**：它有 before.pt/inputs.pt 的首步对照硬门禁；本包不删除这个旧门禁，也不重播其 UNKNOWN transfer。

普通入口已由原源码证明：`FRAMEWORK/tools/hygon_training_recipe.py T09.sft --a100-launch ORIGINAL_LAUNCH.json --asset-map ASSETS.json --run --cards SELECTED --source-root SOURCE_PARENT --output-root NEW_OUTPUT --run-name UNIQUE --lock-root NEW_LOCK --lock-host ACTUAL_HOST`。三个 target 资产的精确映射在本机 result 的 bindings；source-parent 为 `/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-original-source-v1/source`。上述 argv 是 controller 接线合同，不是已准入或已调度命令。

现有 cache loader 明确 `weights_only=False`，仅可消费已有授权、精确 SHA 的原缓存；不得在本包中替换/新造文件或扩大反序列化可信范围。原 before.pt/inputs.pt 仍是原 A100 数值对照的实缺项。独立回归即使未来产生更新，也不能补写原数值 PASS、正式17资格、领域效果或性能。

本机调用（输出必须尚不存在）：

```powershell
python -B build_plan.py --repo ABS_REPO --output NEW_PRIVATE_RESULT.json
```

本轮实际结果在 `local-only/training-hunyuanvideo-cached-native-fallback-plan-v1-original/result.json`，真实设备/更新/新READY均0。root 独占发布、claim、bounded controller 和目标实际执行。
