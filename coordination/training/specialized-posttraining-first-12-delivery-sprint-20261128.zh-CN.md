# 领域模型首批 12 个底座交付冲刺

本轮每个底座保留一个主方向。先兑现已有检查点，再补最短的训练依赖。12 个底座各有一份通过领域提升与新进程加载验收的权重或适配器后，再扩大到 50 个。原 17 项适配、A100 对照和领域增益分别记账。优先级相同的作业，先安排能产出下一份可复核证据的作业。 [50 项候选清单](specialized-posttraining-50-v1.zh-CN.md)

## 12 个底座的下一步

| 底座 | 首交付能力、数据和方法 | 眼下要完成的工序 | 独立验收及失败出口 |
|---|---|---|---|
| Gemma-3-4B-IT | USGS 地震图表读数与单位识别。保留现有语言 LoRA 与 512 序列配方。 | 复用现有 CP25。96 道独立题中，CP25 答对 52 道，底座答对 40 道。对照 CP20 第 21 步的输入、各 rank RNG、LR 和累积边界，补 TextVQA 同题保持性和 COCO CIDEr。TextVQA 已在本机核 2,363 张图、对应 3,982 题。 | 按地震事件分组比较读数和单位。保留当轮盲测封存结果，后续版本换事件。 [领域评分及恢复差异](../../results/bw1000/2026-11-27/specialized-gemma-usgs-cp25-release-audit.json) · [USGS 地震目录](https://earthquake.usgs.gov/fdsnws/event/1/) |
| InternVL3.5-8B-HF | GQA 关系查询与真假判断，原生 HF 图文处理、语言 LoRA。 | 原 CP25 已在新进程逐值复核 504 个 LoRA 张量。先对账已冻结的原 96 图交接，再运行同图底座与 CP25 两臂并封存。 | 查询和判断各 48 题，按图配对。保持原预注册的 ≥3 个百分点且 95% 下界大于零。 [复载证据](../../results/bw1000/2026-11-28/specialized-internvl-4809-cp25-fresh-native-reload-504-lora-exact.json) · [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) |
| LLaVA-OneVision-1.5-8B-Instruct | 降低物体幻觉。CLEVR 场景图监督 LoRA，再评真实照片。 | 八卡已从 CP2 恢复至 CP25。新进程已通过原 eager 入口载入 CP25，504 个适配器张量逐值一致。下一步冻结按原图隔离的盲测，并记录连续训练与恢复的独立对照。 | CLEVR 先记误报与漏报，真实图用 CHAIR、POPE 和 CIDEr。真实图失败后按实际错误选择训练数据，原盲测继续封存。 [原训练与清场](../../results/bw1000/2026-11-28/specialized-llava-5000-cp2-to-cp25-and-5001-clean.json) · [新进程复载](../../results/bw1000/2026-11-28/specialized-llava-5006-fresh-native-cp25-reload-504-exact.json) · [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) · [CHAIR](https://aclanthology.org/D18-1437/) · [POPE](https://aclanthology.org/2023.emnlp-main.20/) |
| MiniCPM-V-4.6 | 商品包装规格、单位与不可读拒答。以现有 Open Food Facts 小批复用原语言 LoRA。 | 完成原 14 对封存输出的独立评分，并按商品家族补可读、遮挡两类训练样本。原生配方继续保留五步预热和对齐模块冻结。 | 报数字与单位联合准确率、不可读误答、有答案误拒。新商品家族进入正式盲测前核图片许可和近重复。 [配对输出封存](../../results/bw1000/2026-11-27/specialized-minicpm-OFF-4495-paired14-sealed-and-4528-clean-post.json) · [数据发布方](https://world.openfoodfacts.org/data) |
| HunyuanVideo 1.5 | 480p 桌面单品转台，ABO 和合格 CC0 自渲染素材。沿用 Q/V LoRA、缓存和 Muon。 | 原训练已有 25 个更新。以合法单卡环境解决新进程设备保护失败，核 216 个 LoRA 张量，然后跑已冻结的三件商品预筛。 | 按商品家族独立扩大样本，测主体一致性、闪烁率、Video CLIP 和失败率。 [25 步证据](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-v3-4560-25-native-updates-and-4561-independent-post.json) · [VBench](https://github.com/Vchitect/VBench) |
| Z-Image-Turbo | 科学插图风格。使用融合原 Turbo adapter 的底座和已训练 Style18 CP25。 | 固定两位评审、生成设置和盲序，按八条提示、每条两种子出 16 对。 | 评形状结构和目标风格同时合格的比例，通用提示单报。如果领域增益不足，下一版换 Met CC0 陶瓷器形和釉色，按馆藏作品家族留出。 [盲评与生成准备](../../results/bw1000/2026-11-27/specialized-zimage-style18-independent-blinding-and-native-render-package-local.json) · [Met Open Access](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access) |
| VGGT-1B | 室内双帧相对位姿。冻 aggregator，仅训练 camera head。 | 优先评分原底座与已有 CP25 的 20 个 TUM 槽，保留一个几何失败。后续用 TartanAir v2 单条室内轨迹验证 NED 位姿、内参与深度双向重投影，再纳入训练。 | 按真实录制序列留出，主报 AUC@30，附 AUC@5/15、旋转和平移误差与失败率。 [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [TartanAir 位姿和相机](https://tartanair.org/modalities.html#raw-camera-pose) |
| Qwen3-VL-8B-Instruct | 街景 OCR 小字和多行问答。用已过原生输入的 26 条 TextVQA 与 15 条 TextOCR 监督语言 LoRA。 | 先对账原八卡请求。原图分组、训练数据身份和放置通过后，以原 BF16、SDPA、4096 配方验证正 LR 参数更新，接续 25 步。 | 同图配对独立 TextVQA，分别报小字、多行、反光和未作答。ST-VQA 只在逐图许可和重叠检查通过后扩充。 [TextVQA](https://textvqa.org/dataset/) · [TextOCR](https://textvqa.org/textocr/dataset/) |
| SDXL 1.0 | 电商单品棚拍。以合格商品和 CC0 自渲染图做 UNet Attention LoRA。 | 两套官方文本编码器和两件素材的原生 VAE latent 已核验。已闭合官方 18 项清单和两个小配置的目标回读。先整套只读挂载并核完整权重，再用同一图、噪声与 timestep 验正 LR 的首个更新，再接 25 步。 | 按未见商品家族测单品完整与背景合格，同提示和种子；扩足样本后同口径报 CLIP Score 和 FID。 [真实文本条件](../../results/bw1000/2026-11-28/specialized-sdxl-4995-two-actual-native-text-conditions.json) · [Poly Haven CC0](https://polyhaven.com/license) |
| Stable Fast 3D | 金属和非金属商品 PBR 材质还原。只更新原 36 个材质张量。 | 已有工具柜与漆木凳两件可复读渲染。先共同通过原冻结特征入口，验证可见点、RGBA 与三项材质真值对应，再做第一次真实更新。 | 留出资产及材质家族，分别报固定视角 LPIPS、albedo/roughness/metallic MAE。 [第二件渲染证据](../../results/bw1000/2026-11-28/specialized-sf3d-painted-stool-constant-nonmetal-cpu-render.json) |
| HY-Motion 1.0 | 步行、急停和原地转向。原 128 帧缓存和 201 维 Flow Matching LoRA。 | ACCAD B03 独立有效帧 21/92，未达到 79 帧，记录为失败。选择另一条具有独立标定的 ACCAD 或 CMU 动作，先产出一条真实缓存并核帧率、接触、旋转和文本条件。 | 保持原 <40 mm 几何门。按表演者与原记录留出，报方向误差、足滑、R@1、MotionFD。 [B03 实测](../../results/bw1000/2026-11-28/specialized-hymotion-accad-b03-41-label-projection-and-independent-calibration.json) · [ACCAD](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) |
| TRELLIS.2 | 单图硬表面重建。CC0 工具和电器，冻其余组件，只训练 1.3B Shape Flow LoRA。 | 先用完整工具柜网格执行官方体素化与 latent 编码，再让原生训练数据集读取同物体的形状与条件，核 1024/2048 token 限额。第一件因形状契约失败时切录音机。 | 按网格及资产家族留出，报 Chamfer-L1、F-score@1% 和导出失败率。 [官方数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) · [CC0 资产](https://polyhaven.com/license) |

## 排队与验收

1. **已有检查点优先验收。** 并行推进 Gemma 恢复定位、InternVL 与 MiniCPM 封存结果、LLaVA 域内配对盲测。HunyuanVideo、Z-Image、VGGT 各推进一项独立复载或评测。接续现有可用权重，先取得可交付结论。
2. **准备五项真实训练储备。** Qwen 街景 OCR、SDXL 商品棚拍、SF3D 材质头、HY-Motion 转向动作、TRELLIS.2 形状流各保留一个工单。数据、模型权重、启动命令、机器放置与有效更新计划均有回执后才进入可直接提交计数。训练位置开出后，仅替换所需的保底作业。已训模型的复载、评分与制备工作单独记录。
3. **一套框架贯穿全部底座。** 使用既有原生后端、冻结模板及优化器；新数据经统一清单绑定底座 SHA、许可证、数据集 SHA、划分组、训练命令、失败计数、检查点和评分器版本。复用商品资产家族划分，分别生成图像、视频、几何和材质的原生输入。升级到偏好优化或 RL 前，先完成同预算 SFT 对照。
4. **达到领域交付才计数。** 冻结主要指标、实际提升门槛、盲集和求解失败口径，再执行原底座与候选对照。公开原生可加载产物、训练和恢复证据、载入命令及独立配对指标。保留原 17 项平台验收和 A100 吞吐、留出集效果的独立分母。
