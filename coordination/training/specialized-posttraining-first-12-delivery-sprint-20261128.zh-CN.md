# 领域模型首批 12 个底座交付冲刺

本轮每个底座保留一个主方向。先兑现已有检查点，再补最短的训练依赖。12 个底座各有一份通过领域提升与新进程加载验收的权重或适配器后，再扩大到 50 个。原 17 项适配、A100 对照和领域增益分别记账。优先级相同的作业，先安排能产出下一份可复核证据的作业。 [50 项候选清单](specialized-posttraining-50-v1.zh-CN.md)

## 并行执行顺序

| 执行流 | 先交出的证据 | 后续接续 |
| --- | --- | --- |
| 检查点兑现 | Gemma 补恢复和原能力保持。InternVL 与 LLaVA 优先产出独立配对评分，MiniCPM 消费原封存对照。 | HunyuanVideo 补复载和三商品预筛，Z-Image 补固定盲评，VGGT 消费原 20 槽。各模型独立推进，先验收的先发布。 |
| 真实更新 | Qwen 复用 41 条原生监督样本，SDXL 复用已核官方权重和两条图文输入，分别获取正学习率下的参数变化与可复载检查点。 | SF3D 并行核金属和非金属两条冻结特征，更新原 36 个材质张量。 |
| 数据与缓存 | HY-Motion 核转向和急停的独立标定与时序。TRELLIS.2 对完整工具柜生成同资产 O-Voxel、形状 latent 与条件图。 | 两条路线各自通过一条原生训练输入实读再扩大，保留源资产、有效样本、失败原因和实际更新数。 |

四个商品方向复用同一资产身份与家族划分。SDXL 渲染单张 RGB，HunyuanVideo 渲染连续相机轨迹，TRELLIS.2 保留源网格和稀疏条件，SF3D 保留同视角 RGBA 与 PBR 真值。在渲染前按物体家族划分训练集和盲测集。已有两件 CC0 商品用于入口和数值核验，正式效果取独立家族。 [Poly Haven 资产许可](https://polyhaven.com/license) · [TRELLIS.2 官方数据流水线](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md)

HY-Motion 首批源候选是 CMU Subject 91 的 trial 3（连续行走转弯）与 Subject 128 的 trial 5（跑、停、再跑），发布方均标为 120 Hz。分别绑定骨架、AMC 和 C3D，验证有效段能覆盖完整缓存。CMU 说明同一表演者可能有多个编号，跨编号仍需核表演者是否独立。 [CMU 数据说明](https://mocap.cs.cmu.edu/) · [转向试次](https://mocap.cs.cmu.edu/search.php?subjectnumber=91) · [急停试次](https://mocap.cs.cmu.edu/search.php?subjectnumber=128)

VGGT 的补充训练先验证 TartanAir V2 官方示例中的 `ArchVizTinyHouseDay`、`easy`、`P000`、`lcam_front`。读取同帧 RGB、深度和位姿，验证相机原图 640×640、焦距 320 像素、NED 坐标转换和两帧互投，再记录实际 224 像素预处理的内参。整条轨迹归入一个集合，TUM 继续承担真实序列盲测。 [TartanAir 轨迹示例](https://tartanair.org/examples.html) · [相机和位姿定义](https://tartanair.org/modalities.html)

## 12 个底座的下一步

| 底座 | 首交付能力、数据和方法 | 眼下要完成的工序 | 独立验收及失败出口 |
|---|---|---|---|
| Gemma-3-4B-IT | USGS 地震图表读数与单位识别。保留现有语言 LoRA 与 512 序列配方。 | 复用现有 CP25。96 道独立题中，CP25 答对 52 道，底座答对 40 道。对照 CP20 第 21 步的输入、各 rank RNG、LR 和累积边界，补 TextVQA 同题保持性和 COCO CIDEr。TextVQA 的 3,166 张官方图已在本机核验，覆盖全部 5,000 题。目标端重解码了原训练集 64 张图，与 3,166 张验证图无完全相同的 RGB 像素。验证图正在分批暂存。 | 按地震事件分组比较读数和单位。保留当轮盲测封存结果，后续版本换事件。 [领域评分及恢复差异](../../results/bw1000/2026-11-27/specialized-gemma-usgs-cp25-release-audit.json) · [USGS 地震目录](https://earthquake.usgs.gov/fdsnws/event/1/) |
| InternVL3.5-8B-HF | GQA 关系查询与真假判断，原生 HF 图文处理、语言 LoRA。 | 原 CP25 已在新进程逐值复核 504 个 LoRA 张量。先对账已冻结的原 96 图交接，再运行同图底座与 CP25 两臂并封存。 | 查询和判断各 48 题，按图配对。保持原预注册的 ≥3 个百分点且 95% 下界大于零。 [复载证据](../../results/bw1000/2026-11-28/specialized-internvl-4809-cp25-fresh-native-reload-504-lora-exact.json) · [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) |
| LLaVA-OneVision-1.5-8B-Instruct | 降低物体幻觉。CLEVR 场景图监督 LoRA，再评真实照片。 | 八卡已从 CP2 恢复至 CP25。新进程已通过原 eager 入口载入 CP25，504 个适配器张量逐值一致。已冻结 16 张此前未用的 CLEVR 验证图，不含答案的输入与 32 槽原生评测入口已在目标端核验。单卡评测 `1103606.5112` 在基础模型加载后的 eager 联合断言失败，32 槽均缺失，独立检查 `1103606.5114` 确认释放。先核模型类、参数数与 dtype、模板、eager 设置和 RoPE 标记各子条件，再发新评测。连续与恢复训练的对照仍待完成。 [盲测终态](../../results/bw1000/2026-11-28/specialized-llava-clevr-val16-5112-eager-base-gate-and-5114-independent-clean.json) | CLEVR 先记误报与漏报，真实图用 CHAIR、POPE 和 CIDEr。真实图失败后按实际错误选择训练数据，原盲测继续封存。 [原训练与清场](../../results/bw1000/2026-11-28/specialized-llava-5000-cp2-to-cp25-and-5001-clean.json) · [新进程复载](../../results/bw1000/2026-11-28/specialized-llava-5006-fresh-native-cp25-reload-504-exact.json) · [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) · [CHAIR](https://aclanthology.org/D18-1437/) · [POPE](https://aclanthology.org/2023.emnlp-main.20/) |
| MiniCPM-V-4.6 | 商品包装规格、单位与不可读拒答。以现有 Open Food Facts 小批复用原语言 LoRA。 | 完成原 14 对封存输出的独立评分，并按商品家族补可读、遮挡两类训练样本。原生配方继续保留五步预热和对齐模块冻结。 | 报数字与单位联合准确率、不可读误答、有答案误拒。新商品家族进入正式盲测前核图片许可和近重复。 [配对输出封存](../../results/bw1000/2026-11-27/specialized-minicpm-OFF-4495-paired14-sealed-and-4528-clean-post.json) · [数据发布方](https://world.openfoodfacts.org/data) |
| HunyuanVideo 1.5 | 480p 桌面单品转台，ABO 和合格 CC0 自渲染素材。沿用 Q/V LoRA、缓存和 Muon。 | 原训练已有 25 个更新。以合法单卡环境解决新进程设备保护失败，核 216 个 LoRA 张量，然后跑已冻结的三件商品预筛。 | 按商品家族独立扩大样本，测主体一致性、闪烁率、Video CLIP 和失败率。 [25 步证据](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-v3-4560-25-native-updates-and-4561-independent-post.json) · [VBench](https://github.com/Vchitect/VBench) |
| Z-Image-Turbo | 科学插图风格。使用融合原 Turbo adapter 的底座和已训练 Style18 CP25。 | 固定两位评审、生成设置和盲序，按八条提示、每条两种子出 16 对。 | 评形状结构和目标风格同时合格的比例，通用提示单报。如果领域增益不足，下一版换 Met CC0 陶瓷器形和釉色，按馆藏作品家族留出。 [盲评与生成准备](../../results/bw1000/2026-11-27/specialized-zimage-style18-independent-blinding-and-native-render-package-local.json) · [Met Open Access](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access) |
| VGGT-1B | 室内双帧相对位姿。冻 aggregator，仅训练 camera head。 | 优先评分原底座与已有 CP25 的 20 个 TUM 槽，保留一个几何失败。后续用 TartanAir v2 单条室内轨迹验证 NED 位姿、内参与深度双向重投影，再纳入训练。 | 按真实录制序列留出，主报 AUC@30，附 AUC@5/15、旋转和平移误差与失败率。 [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [TartanAir 位姿和相机](https://tartanair.org/modalities.html#raw-camera-pose) |
| Qwen3-VL-8B-Instruct | 街景 OCR 小字和多行问答。用已过原生输入的 26 条 TextVQA 与 15 条 TextOCR 监督语言 LoRA。 | 先对账原八卡请求。原图分组、训练数据身份和放置通过后，以原 BF16、SDPA、4096 配方验证正 LR 参数更新，接续 25 步。 | 同图配对独立 TextVQA，分别报小字、多行、反光和未作答。ST-VQA 只在逐图许可和重叠检查通过后扩充。 [TextVQA](https://textvqa.org/dataset/) · [TextOCR](https://textvqa.org/textocr/dataset/) |
| SDXL 1.0 | 电商单品棚拍。以合格商品和 CC0 自渲染图做 UNet Attention LoRA。 | 两套官方文本编码器和两件素材的原生 VAE latent 已核验。目标端已对 18 个官方文件共 13,878,864,542 字节完成只读挂载和 SHA256 核验，两条实际原生图文输入已读取。单卡步骤 `1194900.2664` 在前向前因缺 Diffusers 退出，记录零更新和独立清场。修复挂载后，`1194900.2670` 完成首个真实更新，学习率 0.0001，560 个张量变化，CP1 四个成员独立校验且 `1194900.2671` 清场。单卡 HIP 恢复候选已通过本机参数与源代码验证，连续跑 25 步后可从 CP20 新进程接续 5 步。若原配方指定 CP10，则从 CP10 接续 15 步。实际 GPU 连续训练与恢复待执行。 [HIP 恢复源码资格](../../results/bw1000/2026-11-28/specialized-sdxl-native-hip-full-resume-source-and-local-fixtures.json) | 按未见商品家族测单品完整与背景合格，同提示和种子；扩足样本后同口径报 CLIP Score 和 FID。 [实际整管线与图文输入](../../results/bw1000/2026-11-28/specialized-sdxl-5007-full-pipeline-and-5039-native-imagefolder.json) · [首步与 CP1 证据](../../results/bw1000/2026-11-28/specialized-sdxl-first1-v3-2670-native-update-cp1-and-independent-clean.json) · [Poly Haven CC0](https://polyhaven.com/license) |
| Stable Fast 3D | 金属和非金属商品 PBR 材质还原。只更新原 36 个材质张量。 | 工具柜与漆木凳已在本机同时通过沿用原逻辑的双样本读取器，顺序反转被拒绝。下一步在目标端验证冻结特征、原 36 张量和第一次真实更新。 [双样本读取回执](../../results/bw1000/2026-11-28/specialized-sf3d-cc0-two-object-native-parser-local.json) | 留出资产及材质家族，分别报固定视角 LPIPS、albedo/roughness/metallic MAE。 [第二件渲染证据](../../results/bw1000/2026-11-28/specialized-sf3d-painted-stool-constant-nonmetal-cpu-render.json) |
| HY-Motion 1.0 | 步行、急停和原地转向。原 128 帧缓存和 201 维 Flow Matching LoRA。 | ACCAD B03 独立有效帧 21/92，未达到 79 帧，记录为失败。选择另一条具有独立标定的 ACCAD 或 CMU 动作，先产出一条真实缓存并核帧率、接触、旋转和文本条件。 | 保持原 <40 mm 几何门。按表演者与原记录留出，报方向误差、足滑、R@1、MotionFD。 [B03 实测](../../results/bw1000/2026-11-28/specialized-hymotion-accad-b03-41-label-projection-and-independent-calibration.json) · [ACCAD](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) |
| TRELLIS.2 | 单图硬表面重建。CC0 工具和电器，冻其余组件，只训练 1.3B Shape Flow LoRA。 | 先用完整工具柜网格执行官方体素化与 latent 编码，再让原生训练数据集读取同物体的形状与条件，核 1024/2048 token 限额。第一件因形状契约失败时切录音机。 | 按网格及资产家族留出，报 Chamfer-L1、F-score@1% 和导出失败率。 [官方数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) · [CC0 资产](https://polyhaven.com/license) |

## 排队与验收

1. **已有检查点优先验收。** 并行推进 Gemma 恢复定位、InternVL 与 MiniCPM 封存结果、LLaVA 域内配对盲测。HunyuanVideo、Z-Image、VGGT 各推进一项独立复载或评测。接续现有可用权重，先取得可交付结论。
2. **准备五项真实训练储备。** Qwen 街景 OCR、SDXL 商品棚拍、SF3D 材质头、HY-Motion 转向动作、TRELLIS.2 形状流各保留一个工单。数据、模型权重、启动命令、机器放置与有效更新计划均有回执后才进入可直接提交计数。训练位置开出后，仅替换所需的保底作业。已训模型的复载、评分与制备工作单独记录。
3. **一套框架贯穿全部底座。** 使用既有原生后端、冻结模板及优化器；新数据经统一清单绑定底座 SHA、许可证、数据集 SHA、划分组、训练命令、失败计数、检查点和评分器版本。复用商品资产家族划分，分别生成图像、视频、几何和材质的原生输入。升级到偏好优化或 RL 前，先完成同预算 SFT 对照。
4. **达到领域交付才计数。** 冻结主要指标、实际提升门槛、盲集和求解失败口径，再执行原底座与候选对照。公开原生可加载产物、训练和恢复证据、载入命令及独立配对指标。保留原 17 项平台验收和 A100 吞吐、留出集效果的独立分母。
