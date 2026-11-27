# 十二个底座的领域模型首轮执行清单

目标是先从每个底座各交付一个可加载、在独立数据上实测提升的模型，再增加同底座的其他方向。沿用统一框架的原生训练入口和冻结边界。下面的首步、25 步和开发集分别用于排错、验证连续更新和挑选候选。

## 首批并行执行

| 底座与交付能力 | 数据与训练方法 | 下一个可核验结果 |
|---|---|---|
| Gemma-3-4B-IT · 地震图表读数 | 复用原 USGS CP25 和 512 序列原生图文 LoRA。沿原事件分组，保留已封存 96 题，进一步做 TextVQA/CIDEr 保持性。 | 已有独立盲测 52/96，底座 40/96，差值 +12.5 个百分点。先验交付副本和第 21 步恢复，后续 R120 与原 CP25 分别核算。 |
| VGGT-1B · 室内双帧相对位姿 | 复用原相机头 CP25，冻结 aggregator，224 像素双帧在线提特征。按 TUM RGB-D 录制隔离。 | 用原底座与 CP25 评冻结 household 20 槽，保留 1 个几何不合格槽，报告 AUC@30/15/5。目标产物传输原审批仍适用。 |
| Z-Image-Turbo · 科学图表风格 | 复用 Style18 CP25，按同一融合 Turbo 底座、8 个提示、2 个种子成对生成。 | 先锁两名独立评审和盲序，再组织 16 对领域与通用对照。领域主指标为风格和结构同时合格率。 |
| Qwen3-VL-8B-Instruct · 街景 OCR 问答 | TextVQA 训练问答加 TextOCR 读字和位置辅助监督。沿用已过原生处理的 26 条问答和 15 条 OCR 样本。 | 先查原预约和唯一接续窗口，在原八卡 BF16 配方核前两个真实更新和参数差，再按图像去重、按小字等难度分层做官方计分。 |
| InternVL3.5-8B-HF · 物体关系问答 | 官方 GQA balanced train 的 `types.semantic == "rel"`，短答案 SFT。每个来源组只取一条，按图像 ID、来源组和内容摘要排除开发及盲测交叉。 | 原 A100 25 步是 USGS，供复用原生配方和入口。先从已核的关系标注绑定 64 条逐图准入样本，再用真实训练图验 HF processor、标签和 8 卡首次正学习率参数差。独立盲测报关系题准确率、官方 consistency。 |
| HunyuanVideo 1.5 · 桌面商品转台 | 已有一个 ABO 真实资产的 9 帧 848×480 原生缓存。保留 Q/V LoRA、480p 原生入口及 Muon，按 SKU 家族留出。 | 目标容器的原生入口及优化器钩子已通过 CPU 验证。独立单卡候选的下一证据是实跑前后向、正学习率及参数变化。评主体一致性、闪烁与 Video CLIP。 |
| SDXL 1.0 · 商品棚拍 | ABO 目录主图或 CC0 配对商品渲染，使用图中实际可见的 caption。UNet Attention LoRA，冻结文本编码器和 VAE。 | 并行做一张训练图的身份与预处理验证，以及原同输入反向差异定位。按商品家族/背景模板隔离，报棚拍适用率、CLIP Score、FID 与分母。 |
| MiniCPM-V-4.6 · 包装规格和不可读拒答 | 首选 ABO 包装正面图加人工读出的数值及单位，补许可明确的可控渲染。冻结视觉和对齐模块，保留 5 步预热。 | 原生读入一条可读及一条不可读任务，核监督 mask，再验证首次参数变化。真实未见 SKU 盲测报数字加单位准确率及误答率。 |
| LLaVA-OneVision-1.5-8B-Instruct · 减少物体幻觉 | 使用能核图像许可的 COCO captions 和目标标注做 grounded caption SFT，许可或配对不足时用自渲染物体清单作训练替补。保留 eager attention。 | 核一张图的 caption 实体与存在标注并过原生 processor；并行闭合目标权重分片。以 CHAIR 与 CIDEr 成对测试，POPE 可单独报告物体存在问答。 |
| TRELLIS.2 · 硬表面单图重建 | 用完整 ABO 网格和同物体相机渲染，通过原 O-Voxel/Shape Flow 数据入口。ABO 配对不足时使用许可明确的 CC0 完整网格。 | 首个目标是一条真实原生 latent，保留 1024/2048 token 上限和 Shape Flow LoRA。未见资产报 Chamfer-L1、F-score@1% 及失败。 |
| Stable Fast 3D · 商品材质恢复 | 先做 Poly Haven 不透明 `metal_tool_chest` 固定视角。RGBA、前景掩码、相机与三类 PBR 真值同视角，原图集独立核值。 | 由 1K 原生图集核验推进到一条有可见像素监督的输入，再核原生冻结特征及 36 个材质张量。按物体与材质族留出，报 LPIPS 和三类 MAE。 |
| HY-Motion 1.0 · 步行、急停、转向 | 主线使用已检查源端运动学的 ACCAD，按表演者留出。CMU Graphics Lab 为独立替补。原 Flow Matching LoRA 和原生文本编码。 | 先闭合一条真实连续 128 帧、201 维训练缓存，对账时间轴、关节、足部接触及旋转。独立留出报 R@1、MotionFD、足滑和方向误差。 |

## 资源接续与验收

- 检查点路线先闭合独立评分及可加载产物。Qwen、HunyuanVideo 紧接着验证真实更新。InternVL、LLaVA、SDXL、MiniCPM、SF3D、TRELLIS 和 HY-Motion 的数据与原生入口并行准备。其中 SF3D 的同视角配对和 InternVL 的数据切片先走 CPU。
- 训练储备按具有新作业身份、数据、权重、启动入口、正学习率更新计划和实际可放置资源的任务计数。就绪一项即接替对应节点的保底。五个 VLM 的原八卡性能和 17 项基础配方仍各自按 A100 验收。
- 开发集用于冻结训练长度和指标阈值。冻结后同输入对比原底座与候选，保留缺失、生成失败和未通过的分母。首轮领域交付至少有可加载文件、实际更新、盲测提升和新进程复验。Gemma 原 96 题已揭盲，后续版本使用新的独立事件组。
- 共享商品资产的目录照片、网格、像素材质和视频分别对应各自原始文件。用物体家族、原图身份和内容摘要阻断跨用途的训练与评测交叉。COCO、GQA 逐图记录权利来源。

## InternVL 数据入口

`build_gqa_relation_train.py` 消费固定 SHA256 的官方 `train_balanced_questions.json`、逐图媒体许可清单和只含身份的留出排除清单。准入图像条目记录相对路径、原图内容 SHA256、来源组、来源网址和许可。留出清单记录图像 ID、来源组和原图内容 SHA256，开发与最终留出的并集即可，不进入答案。按固定种子排序后每组最多取一条，训练行与出处回执仅保存在实验工作区。目标原生处理和真实更新单独验证。

已下载官方 ZIP 中约 109 MB 的唯一训练成员。流式校验原压缩流和解压后标注，覆盖 943,000 条训练问题，其中 438,964 条是关系问答，涉及 58,206 个图像 ID。用 `--questions-compressed-sha` 和 `--questions-uncompressed-bytes` 可直接消费压缩成员。逐图核验和留出绑定后才形成真实训练行。 [原始训练成员盘点](../../results/bw1000/2026-11-27/specialized-internvl-GQA-original-member-stream-inventory.json)

用 Visual Genome 的 COCO/Flickr 双身份和 COCO 逐图许可筛出 4,331 个无歧义的 CC BY 2.0 元数据候选。首批 8 个实际 JPEG 来自 Stanford GQA 官方图像包，均通过 ZIP CRC、JPEG 读回及尺寸核对。按来源组固定 4 个训练、2 个开发预留和 2 个独立预留，生成 4 条真实 Swift 格式输入。下一项是目标环境原生 processor 验证。正式发布前补齐权利人署名，正式评测另行冻结。 [原图与训练输入实查](../../results/bw1000/2026-11-27/specialized-internvl-GQA-coco-vg-original-jpegs-and-four-training-rows.json)

[GQA 官方数据和字段](https://cs.stanford.edu/people/dorarad/gqa/download.html) · [GQA 官方评价](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) · [TextOCR 官方数据与 TextVQA 图像对应](https://textvqa.org/textocr/dataset/) · [COCO 逐图许可说明](https://cocodataset.org/dataset/termsofuse.htm) · [CHAIR 原论文](https://aclanthology.org/D18-1437/) · [POPE 原论文](https://arxiv.org/abs/2305.10355)

[USGS 素材权利与署名](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits) · [ABO 官方数据](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [Poly Haven 许可](https://polyhaven.com/license) · [TRELLIS.2 官方训练数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) · [VBench 指标定义](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Huang_VBench_Comprehensive_Benchmark_CVPR_2024_supplemental.pdf) · [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [ACCAD](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) · [CMU Graphics Lab 动捕](https://mocap.cs.cmu.edu/)

[原 12 底座执行单](specialized-posttraining-12-accelerated-execution-20261127.zh-CN.md) · [Gemma 发布证据](../../results/bw1000/2026-11-27/specialized-gemma-usgs-cp25-release-audit.json) · [HunyuanVideo 训练入口目标容器证据](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-4313-import-failure-4314-bound-runtime-CPU.json) · [SF3D 图集验证](../../results/bw1000/2026-11-27/specialized-sf3d-polyhaven-metal-tool-chest-native-atlas-target-readback.json) · [InternVL 数据入口](../../code/bw1000/training-domain-pilot-v1/build_gqa_relation_train.py)
