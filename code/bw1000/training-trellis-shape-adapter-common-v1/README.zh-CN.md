# TRELLIS.2 训练适配器生成入口

统一框架候选 `3f37d53b1efcea039d2ed808e3b5a1a1916cd596` 基于已独立导入的 `9338c7ebedf7ed8886887949956042c912b36d6b`。在该父提交上用 `git apply --index --unidiff-zero framework.patch`，然后用 `git write-tree` 对照 `SOURCE_PROOF.json` 的树。裁剪函数与已验证四图时的实现保持同一源码摘要。

从 `gencuda infer --model trellis-2-4b` 使用训练后适配器时，在推理配置中设置 `shape_flow_adapter`，包括 `adapter`、`adapter_sha256`、`training_manifest`、`training_manifest_sha256` 和 `effective_updates`。训练清单使用原 `gencuda train --model trellis-shape-flow-1.3b` 的完成态 `manifest.json`，并绑定原 `adapter-final.safetensors`。无适配器配置的初始组沿用常规生成入口。海光和 A100 的训练后组各用自己的原清单，配对时要求相同有效更新数。

实际生成在 GPU 初始化前核对适配器和清单。到 Shape Flow 阶段加载原训练底座，验证 24 个 FP32 适配器张量，复用训练使用的 rank 2 / alpha 4 实现，并在首次采样前加载。加载保留 CPU 随机数状态，生成记录保存 adapter、训练清单和底座摘要以及有效更新数。适配器加载计时单列。

本轮先做一次零 GRES 容器检查，资源固定为父作业 `1103606` 的 `f13r4n05`，2 CPU / 2048 MiB，无模型权重和业务数据。冻结源码集与启动器进入 Git 后才允许提交。发生连接中断时按原 `trellis-shape-adapter-cpu-v1` 标签、私有 intent/收据和编号步骤对账。该测试覆盖真实 Torch/SafeTensors 载入逻辑及小型输出计算，真实 TRELLIS 预测仍需模型组件、A100/海光匹配更新的适配器和配准。
