# T07 Flow-GRPO 原生奖励输入验证

在统一框架的原生 NumPy 颜色奖励入口上复核已有的科学报告样式图。实际读取 18 张 512×512 PNG，共 669,041 字节，覆盖 12 张训练图和 6 张开发图。全部通过，耗时 0.797 秒。新独立来源、盲测读取、策略 rollout、优化器更新和新增可派发训练均为 0。

## 实际入口

冻结归档 `code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz` 中，`generation/src/gencuda/backends/zimage_flow_grpo.py` 第 132 至 151 行定义了 `image_prompt_scores`。本包按原 SHA 校验归档和模块，并用 `native_reward.load_native` 直接提取完整函数 AST。原 `run_zimage_flow_grpo` 第 560 行通过 `image_prompt_reward` 使用此函数的分数。

`score_batch4(native, images, frozen_target_rgb)` 要求恰好 4 张同尺寸完整图、宽高至少 2 和有效 RGB 目标。此次在 4 张真实参考图上核验批量调用的一致性，另外核验 0、3、5 张输入均被拒绝。目标环境的 256×256 原生缩放、模型、VAE 和实际策略调用仍待验证。

## 数据与结果

沿用 `training-zimage-USGS-style18-dataset-v1-original/dataset` 的原分流，校验元数据、图片 SHA 和 RGB 像素 SHA。训练覆盖 3 个来源组和 72 个事件，开发覆盖 2 个来源组和 36 个事件，组、事件及 RGB 像素 SHA 交叉均为 0。图片由项目基于 [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) 的公开数值生成。USGS 自产数据的许可边界见其 [版权与署名说明](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)。

在读取图像前冻结诊断目标 RGB 为 `[0.65, 0.2, 0.15]`，沿用 backend 默认值。原正式 Flow-GRPO 启动配置覆盖值的正文尚缺。原启动 SHA 和此次诊断 recipe 见 `PLAN.json`。

逐张执行原图和水平镜像奖励，并执行一次四图批量调用。原图奖励的总体标准差为训练 0.012237、开发 0.015431。镜像翻转图中文字和布局，奖励最大变化为 1.3411e-6，只有 4/18 张落入预先固定的 1e-7 诊断带。此数值刻画了颜色奖励对镜像的敏感度，图表文字、数值和科学语义应独立评分。

本轮完成 CPU 奖励输入这一项前置。实际训练还需核对同一 Turbo 融合底座、目标原生输入、固定噪声、4 条 rollout 各 8 步、old/new log-prob、ratio、advantage 与策略更新。原 SFT 失败和未明目标状态均据原记录保留。原请求 `c8b98ee04e8f4e93b7b78805ae115929` 的独占本机结果已封存。
