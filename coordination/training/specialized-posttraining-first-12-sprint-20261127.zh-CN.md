# 首批 12 个领域模型执行计划

每个底座先交付一个能重新加载、在独立盲测有可复验提升的权重或适配器，再扩展至 50 个领域模型。所有线路使用同一训练框架的模型原生入口和冻结边界。领域验证与 17 项基础训练验收分别记录。 [50 项长期规划](specialized-posttraining-50-v1.zh-CN.md)

| 顺序 | 底座与能力 | 下一项可执行验证 | 独立盲测指标 |
|---:|---|---|---|
| 1 | Gemma-3-4B-IT · 地震图表读数 | 原 USGS CP25 的 25 次更新、适配器和新进程加载已核实。后续八卡作业定位到首批标签形状冲突。按原 Swift 配方复验标签选择及损失语义，再安排真实更新。并行推进第 21 步恢复和通用保持性。 | 已封存 8 个事件组 96 题。领域 52/96，原初态基线 40/96，仅问题对照 23/96。调参后另开独立事件组。 |
| 2 | VGGT-1B · 室内 RGB 双帧位姿（#48） | 复用原 3493 CP25 和已通过的独立 CPU 重载。核实目标送达和实际准入后，用原底座和 CP25 两臂评分。 | TUM household 录制独立留出，AUC@30 主报，AUC@5/15 及缺失、失配单列。保留已冻结 20 槽中不合格的 1 槽。 |
| 3 | Z-Image-Turbo · 科学图表风格（后续 #30） | 使用已有 Style18 连续 CP25 和 Turbo 融合底座，固定新提示、参考图和种子后成对生成。原恢复第 17 步漂移独立定位。 | 提示、媒体和来源组与 Dev12 隔离，主报风格与结构同时合格率，附 CLIP 和通用提示回归。 |
| 4 | InternVL3.5-8B-HF · 关系问答（#11） | 核对 A100 已有两臂的实际训练清单，符合 GQA 方可复用，否则先制备原生 HF GQA 训练批。海光原 USGS 未决请求继续对账。 | GQA accuracy、consistency。训练和开发按原图及关系模板隔离，单列关系题。 |
| 5 | Qwen3-VL-8B-Instruct · 街景文字问答（#01） | 接续已通过原生处理的 41 条街景输入，先清原预约，再做含正学习率的首两步和真实参数差核验。 | TextVQA 官方计分。按原图排除 TextOCR 与正式保持性评测交集，分别报告小字、多行及反光。 |
| 6 | HunyuanVideo 1.5 · 桌面单品转台视频（#38） | 首个 ABO 转台样本已生成并核对 9 帧、848×480。按原 VAE 和文字编码器生成原生缓存后，复用已实跑的 480p Muon 和 Q/V LoRA。 | SKU 家族和所有帧整组留出，主体一致性为主，附闪烁率、Video CLIP 及对应 VBench 维度。 |
| 7 | SDXL 1.0 · 商品棚拍（#26） | 以已有同输入结果继续找反向梯度差。同时冻结 CC BY 素材、caption 和 UNet Attention LoRA 输入。 | 按 SKU 家族和背景模板留出，报告棚拍适用率、CLIP Score、FID 样本量和失败数。 |
| 8 | MiniCPM-V-4.6 · 包装规格读取（#21） | 用在包装图中实际可见的数值、单位和不可读标签做原生训练批。冻结视觉和对齐模块，保留五步预热，验首个改变参数的步。 | 未见 SKU 家族的数值与单位联合准确率，不可读项误答率单列。 |
| 9 | LLaVA-OneVision-1.5-8B-Instruct · 减少描述幻觉（#16） | 核齐目标权重、processor 和初始 adapter；用许可可核的图及目标标注构造原 OneVision 训练批，先做 SFT。 | 按原图、相机和近重复分组，报告物体幻觉率、CIDEr 和外部来源错误率。 |
| 10 | TRELLIS.2 · 硬表面单图重建（#42） | 冻结完整 mesh、相机和配对渲染，验证原生 O-Voxel/Shape Flow 缓存。继续解决海光原首步数值失败，只训练原 1.3B Shape Flow LoRA。 | 商品家族与 mesh 身份双重留出，Chamfer-L1、F-score@1% 和导出失败率。 |
| 11 | Stable Fast 3D · 商品 PBR 材质（#45） | 复用已有 RGBA、相机、固定视角和表面材料参考。核齐权重与原 36 个材质张量后验证真实更新。 | 物体、材质族及光照留出，三类材质 MAE 及固定视角 LPIPS。 |
| 12 | HY-Motion 1.0 · 步行、急停和转向（#34） | 先闭合一条许可明确的真实 128 帧、201 维缓存和原生文本编码。保留 Flow Matching LoRA 的原目标层。 | 表演者与原录制整组留出，方向误差、足滑、R@1 和 MotionFD。60 帧资源只进入对应独立配方。 |

Gemma 与 Z-Image 首批采用已有真实训练方向。原计划的 VizWiz（#06）和商品纯背景（#30）继续保留。Gemma、VGGT、Z-Image 先消费已有权重，减少重新训练、下载和加载。 [Gemma 独立评分](../../results/bw1000/2026-11-27/training-gemma-usgs-confirmatory96-original3682-first-offline-score.json) · [VGGT 检查点重载](../../results/bw1000/2026-11-26/training-vggt-3493-cp25-original-native-3712-cpu-reload-pass.json) · [Z-Image 开发评分](../../results/bw1000/2026-11-22/training-zimage-style18-1149-openai-clip-dev12-terminal.json)

Gemma 后续八卡作业在全部 8 个 rank 的首批输入处触发标签形状核对，实际更新为零。HunyuanVideo 新样本已有完整帧序和逐帧摘要，原目标模型目录尚缺缓存编码依赖。 [Gemma 八卡终态](../../results/bw1000/2026-11-27/training-gemma-r120-4177-compute-source-ready-first-batch-eight-rank-terminal.json) · [HunyuanVideo 样本及编码依赖](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-first-nine-frame-media-and-encoder-asset-gap.json)

## 共享数据和并行排程

1. 商品线服务 MiniCPM、SDXL、Z-Image、HunyuanVideo、TRELLIS、SF3D。ABO 采用 CC BY 4.0，Poly Haven 资产采用 CC0。目录图、视频序列、完整网格和 PBR 真值分别验实际可用性。自渲染保留原始资产、帧轨迹、相机和设置。 [ABO 官方入口](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [Poly Haven 许可](https://polyhaven.com/license)
2. 文字视觉线先供 Qwen 和 MiniCPM 复用图像身份、近重复及正式保持性评测排除工具。COCO 标注和图片许可分别登记。USGS 素材核对制作方与第三方限制。 [TextVQA 作者说明](https://huggingface.co/datasets/facebook/textvqa/blob/main/README.md) · [GQA 官方评价](https://cs.stanford.edu/~dorarad/gqa/evaluate.html) · [COCO 条款](https://cocodataset.org/dataset/termsofuse.htm) · [USGS 版权说明](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)
3. 规划几何与动作时，先闭合原位姿、采集序列或缓存。新盲测在评分前固定主指标、提升门槛、分组、样本量和失败处理。都对照原底座。新增方法时另加同数据、同预算 SFT 对照。 [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [ACCAD 动作数据](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) · [TRELLIS.2 数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md)
4. 同时推进三类任务。已有检查点做重载、盲测和恢复；已核数据做原生首步及 25 步训练；其余做数据及缓存的 CPU 制备。就绪储备只计已核权重、数据、入口、真实更新计划和放置的独立作业，随后接替对应保底。日报逐底座记录素材组、真实更新、独立盲测、可加载产物及正式交付。

已对官方 ABO 前缀实做来源分组。376 条记录形成 216 组，10 个组含多条 listing，最大一组有 90 条复用同批图像。30 条关联已索引图像、15 条关联已索引 mesh，仅 1 条同时关联主图和 mesh。先按此结果扩展配对资产，再在完整候选及内容去重后冻结训练和盲测划分。 [来源分组工具](../../tools/project_abo_source_groups.py)
