# 首批 12 个领域模型加速执行单

首个交付目标是 12 个底座各交付 1 个能重载、在独立盲测有实测提升的适配器或权重，然后扩展到 50 个。统一框架复用原生模板、组件、训练冻结边界和检查点接口。执行顺序以最早能改变交付判断的证据为准，已有检查点优先补评测和恢复，新方向先跑真实正学习率更新。

[原首批执行计划](specialized-posttraining-first-12-sprint-20261127.zh-CN.md) · [50 项路线](specialized-posttraining-50-v1.zh-CN.md)

| 执行顺序 | 底座和首个领域能力 | 数据与训练入口 | 本轮完成条件和独立评测 |
|---|---|---|---|
| 先验收 | **Gemma-3-4B-IT** · 地震图表读数 | 原 USGS 真实图表和原生 `gemma3_vision` LoRA CP25。沿用原 512 序列上限。 | 复用已重载 CP25，优先闭合第 21 步恢复、TextVQA/CIDEr 通用保持和交付文件验证。原独立盲测 52/96 对 40/96，按地震事件分组。R120 新训练继续核标签，独立记录。 |
| 先验收 | **VGGT-1B** · 室内 RGB 双帧位姿 | 原相机头 CP25、TUM RGB-D 新录制、原生在线特征提取。 | 在完成已有传输准入后，底座与 CP25 同跑几何合格的 19 对，每臂保留 20 槽分母。主报 AUC@30，另报 AUC@5/15 及失败。 |
| 先验收 | **Z-Image-Turbo** · 科学图表风格 | 原 Style18 连续 CP25，对比同一 Turbo 融合底座。 | 冻结两名评审的独立 A/B 顺序后，原生入口配对渲染 16 对。先用 4 个领域及 4 个通用提示验证可评分性，主报风格和结构同时合格。正式盲测阈值在下一套留出生成前冻结。 |
| 首批真实更新 | **Qwen3-VL-8B-Instruct** · 街景 OCR 问答 | 已通过原生处理的 41 条街景训练行，用原模板、8 卡 BF16、每卡 1、累积 4 和原 LoRA。 | 原 v5 请求先按原身份对账。可准入时验证含正学习率的 2 次更新及参数差，再进入 25 步。TextVQA 官方评测，按原图剔除 TextOCR 交集并报小字、多行、反光。 |
| 首批真实更新 | **InternVL3.5-8B-HF** · 物体关系问答 | GQA balanced train 中许可明确图像及关系标注。若已有 A100 作业身份吻合则复用参考，否则先制备原生 HF 小批。 | 读入原图后验证原生标签、多图预处理和首个正学习率步。盲测按图像和关系模板分组，报告 GQA accuracy 与 consistency。 |
| 首批真实更新 | **SDXL 1.0** · 商品棚拍 | ABO 无显著水印的目录主图及实物描述，补合格 CC0 自渲染。冻结文本编码器和 VAE，原 UNet Attention LoRA。 | 先复核已有反向差异，再进行同图像、噪声、timestep 的真实更新。按商品系列及背景模板留出。棚拍适用率为主，附 CLIP Score、FID 及样本量。 |
| 首批真实更新 | **MiniCPM-V-4.6** · 包装规格与不可读拒答 | ABO 中包装正面照片及可见规格，少样本时用可控授权标签渲染扩充，真实照片做独立外测。原生模板、视觉及对齐模块冻结。 | 图像可见规格与目录字段逐条核对。保留 5 步预热，首个正学习率步验证参数实际变化。按 SKU 系列验数值与单位联合准确率、误答率。 |
| 首批真实更新 | **LLaVA-OneVision-1.5-8B-Instruct** · 降低物体幻觉 | COCO captions 与目标存在标注逐图核许可，先原 OneVision SFT。数据准入不足时，自渲染带完整物体清单的场景，保留真实外测。 | 实际 processor、eager attention、模型与初始适配器固定后做原生首步。物体幻觉率、CIDEr 和外部图像事实错误率。偏好实验另立同预算 SFT 对照。 |
| 并行补数据 | **HunyuanVideo 1.5** · 桌面单品转台 | ABO 已制备九帧 848×480，原生 VAE 已生成并复核 1 条 latent。同一 caption 已通过原生视频分词和空字形分支。沿用 480p、Q/V LoRA 和 Muon。 | 用核验过的本地 Qwen 权重实际编码，按原缓存格式配对并核读，再做真实更新。按资产家族留出，评主体一致性、闪烁率、Video CLIP 和指定 VBench。 |
| 并行补数据 | **TRELLIS.2** · 硬表面单图重建 | 优先取完整 ABO 网格；配对不足时用 Poly Haven CC0 硬表面完整网格自渲染。使用原 Shape Flow LoRA，保留 1024/2048 token 上限。 | 一件真实资产同源配对图、网格、相机和原生缓存，再过首步。逐资产验 Chamfer-L1、F-score@1% 和导出失败。 |
| 并行补数据 | **Stable Fast 3D** · 商品 PBR 材质 | 先复用 WaterBottle、Corset 训练样本，BoomBox 留开发。Poly Haven 完整 PBR 图补新物体；ABO 透明对象须先验证有效单表面掩码。 | 原生 reader 读入一条真实训练行并记录三种材质标签和可见掩码。仅训练 36 个材质张量；按物体与材质族留出，评固定视角 LPIPS 和各材质 MAE。 |
| 并行补数据 | **HY-Motion 1.0** · 步行、急停和原地转向 | 优先复用已通过源侧 FK 的 ACCAD 片段。同一表演者的中立姿态、C3D 和接触资料用于校准目标骨架；CMU 为独立候选。 | 先在一条训练片段上验证完整 128 帧、201 维目标、异常旋转和足滑；留出表演者不参与修正。后续评方向误差、足滑、R@1 和 MotionFD。 |

## 派发和验收顺序

1. **先验收的并行推进。** Gemma、VGGT、Z-Image 各自复用现有检查点。Gemma 的独立 96 题已有实测收益，正式交付仍受恢复和通用保持约束。VGGT 跟踪原目标准入，Z-Image 优先落实盲评人及评分隔离。
2. **首步确认后，整批交给训练。** 一项候选只有在数据、权重、模型原生入口、明确的正数更新计划及资源均已核验时，才计入 5 项训练储备。每次验收记录实际非零参数更新、全分母及退出原因。已有原请求未决时核对同一请求和调度记录，解决冲突后才排新的明确范围。
3. **共享数据生产，按底座交付。** ABO 的图像、视频/网格、PBR 真值来自不同字段，每条样本都记录媒体匹配、出处和许可。目录图片可用不代表有对应网格；物体网格可用不代表透明表面有无歧义的像素材质真值。Poly Haven 可补真实配对，现有 ABO 下载继续用去重后的资产家族清单。 [ABO 官方数据](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [Poly Haven CC0 条款](https://polyhaven.com/license)
4. **先固定评测，再看结果。** 独立盲测按事件、图像、物体家族、录制或表演者隔离。训练期间可继续用开发集调参。正式交付需新进程加载、预先冻结门槛和相对于同底座的有效独立盲测提升，保留原分母、失败、底座和对照。正式交付率和 17 项基础训练的 A100 指标分别归档。

[USGS 素材版权与署名](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits) · [GQA 官方下载](https://cs.stanford.edu/~dorarad/gqa/download.html) · [TUM RGB-D 数据](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [COCO 图片及标注条款](https://cocodataset.org/dataset/termsofuse.htm) · [ACCAD 动捕原始入口](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data)

[Gemma CP25 交付审计](../../results/bw1000/2026-11-27/specialized-gemma-usgs-cp25-release-audit.json) · [Gemma 目标容器预处理验证](../../results/bw1000/2026-11-27/training-gemma-r120-original4251-posix-ipc-v2-independent.json) · [VGGT CP25 独立重载](../../results/bw1000/2026-11-26/training-vggt-3493-cp25-original-native-3712-cpu-reload-pass.json) · [Z-Image 盲评包](../../results/bw1000/2026-11-27/specialized-zimage-style18-independent-blinding-and-native-render-package-local.json) · [HunyuanVideo 原生 latent](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-native-vae-original4208-one-latent-clean-post.json) · [HunyuanVideo 文本条件核验](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-original4271-native-tokenization-and-empty-glyph.json)
