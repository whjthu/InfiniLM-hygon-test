# 12 个底座的 50 个定制模型规划

领域模型是独立交付目标。用这 12 个底座分别进行领域后训练，先交付覆盖全部 12 个底座的第一批模型，每个底座至少 1 个，再滚动交付到累计 50 个具备明确能力的领域模型。每个交付件包含可独立加载的权重、适配器或头权重，以及底座版本、数据清单、评测证据和调用示例。同一能力的不同种子、对照与较早检查点归入同一个模型条目。

## 进入训练的条件

- 所有方向归入同一训练框架，沿用原 17 项用例的模型入口和训练模块。统一数据身份、训练配方、运行状态、checkpoint、评测和交付清单。不同模型继续使用原生模板、处理器和冻结边界。新损失或新输入合同单独建立对照。基础训练需先通过同输入首步、25 步连续更新和断点恢复。
- 数据清单记录原始链接、版本、样本 ID、用途与再分发条件、近重复分组、处理代码和 train/dev/blind 划分。公开可读取和可以用于训练、允许交付衍生权重分别核实。缺证据的数据只进入备选登记。COCO 和 Open Images 的图片条款逐项核验。 [COCO 使用条款](https://cocodataset.org/dataset/termsofuse.htm) · [Open Images V7 说明](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- 所有行都比较同一底座、同一推理参数下的原模型和定制模型。引入 RL、偏好优化或新损失时，增加同数据、同主体计算预算的常规训练对照；单独记录生成偏好对或 rollout 的成本。下载、处理、训练、留出评测和验收分开计数。
- `★` 是首批 12 个领域交付方向，每底座 1 个。可立即提交的训练作业以其具体阶段的权重、数据、入口、正数更新计划及资源闭合为准。领域长训前仍须通过首步数值与恢复门槛。以至少 5 个可直接提交的真实训练作业为排队目标，队列不足时显示实际数量和阻塞，保留已合格的低优先级回归和硬件任务补位。

## 50 个能力目标

表中测试对象只在评测使用。公开评测集如果与训练来源共享图片、视频、原始记录或物体资产，先做交集剔除。带“自建盲测”的目录在训练前冻结且隔离维护。

### Qwen3-VL-8B-Instruct

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 01 ★ | 街景文字定位并作答 | TextVQA train + 合格 TextOCR 子集，答案 LoRA SFT，加文字定位辅助输出 | TextVQA 官方 test 的 VQA accuracy，另报多行、反光、小字分桶 |
| 02 | 文档字段抽取为 JSON | 自建许可模板文档与可用 DocVQA train，以原生图文模板监督答案、字段及拒答 | 未见文档发行方的字段 micro-F1、JSON 合法率，DocVQA ANLS 外部对照 |
| 03 | 图表比例、同比和趋势问答 | 世界银行可用开放序列自渲染图表，有准入依据时补 ChartQA train，SFT 监督结论与证据坐标 | ChartQAPro 来源去重子集，官方计分，另报数值容差准确率 |
| 04 | 仪表盘和指针读数 | 自建仪表渲染，按仪表系列与拍摄设备留出，监督读数和单位 | 未见设备的量程归一化 MAE、单位正确率 |
| 05 | 环境标识双语问答 | 合规 Open Images/USGS 标识图与人工双语标注，成对 SFT | 另一来源的自建标识盲测，双语语义准确率与语言一致性 |

### Gemma-3-4B-IT

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 06 ★ | 低质量照片的谨慎答复 | VizWiz-VQA train，答案与不可回答联合 SFT，dev 校准拒答门槛 | VizWiz 测试 VQA accuracy 与不可回答 AP，另报有答案项误拒率 |
| 07 | 商品属性短答 | ABO 商品图与结构化属性，限制题量与答案长度，属性 SFT | 未见商品及近似款系列，属性 macro-F1 |
| 08 | 票据关键字段识别 | 自建不含真实个人信息的票据，短字段与空字段 SFT | 未见出票方版式，金额准确率、字段 F1 |
| 09 | 样本是否清晰可用 | VizWiz 可用性标注和有明确阈值的模糊或遮挡变换，SFT 加成对偏好候选 | 自建相机、设备与场景盲测，不可用 AP、误拒率 |
| 10 | 一句话无障碍图片描述 | VizWiz-Captions train 与许可 COCO 图，默认 512 序列约束下 SFT | VizWiz-Captions 隐藏评测 CIDEr 和人工事实错误率 |

### InternVL3.5-8B-HF

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 11 ★ | 基于关系的多步图像问答 | GQA balanced train，保留原生 HF 处理器，用问题和关系证据 SFT | GQA 官方测试 accuracy、consistency，另报关系题分桶 |
| 12 | 科学示意图理解 | 使用条件合格的 ScienceQA/AI2D 训练样本，优先自建带已知概念的图与 QA，监督解释和结论 | 按概念和图源留出，科学图题 accuracy；MathVista 科学子集作外测 |
| 13 | 遮挡与密集场景计数 | 合规 Open Images 框标注与程序合成遮挡场景，计数和证据框 SFT | 未见图源的人工密集计数盲测，精确计数率、计数 MAE |
| 14 | 细节证据定位 | 条款合格的 Flickr30k Entities 与授权图片短语框，输出规范化坐标序列 | 按原始图隔离的盲测 Acc@IoU0.5 和无目标误报率 |
| 15 | 复杂科学图表证据链 | 许可明确的开源科研图和程序渲染曲线，证据区域、读数、最终答案 SFT | 按论文来源留出，答案准确率和证据 IoU |

### LLaVA-OneVision-1.5-8B-Instruct

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 16 ★ | 场景描述少幻觉 | 许可 COCO Captions 与目标存在标注，先 SFT，之后对仅改变物体事实的成对答案验证偏好优化 | COCO 留出 CIDEr、物体幻觉率，外部来源人工检查 |
| 17 | 货架位置与数量 | 开放许可商品图和自建货架拼装，遵照原 OneVision 格式训练问答 | 未见商品家族与货架布局，位置准确率和计数 MAE |
| 18 | 工程示意图诊断 | 自建细分零件图和关系标注，步骤证据 SFT | 按部件族和图模板留出，部件关系 macro-F1 |
| 19 | 一图多格的事件顺序 | 以授权同源帧或自有动画制作单张分镜拼图，用原单图入口做序列 SFT | 整段视频和动作类别组合隔离，顺序准确率与反转错误率 |
| 20 | 图片文字复述 | TextCaps train，拼写精确字段及可见性标记 SFT | TextCaps 官方留出 CIDEr、文字实体 F1 |

### MiniCPM-V-4.6

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 21 ★ | 包装规格与用量读取 | 许可 ABO 商品图加 TextVQA 包装子集，答案、单位、不可读标注 SFT | 未见 SKU 家族的包装盲测，数值单位联合准确率 SKU 分组统计 |
| 22 | 小屏幕和表格截图录入 | 以结构化数据自行生成授权 UI 截图，行列与单元格关系 SFT | 页面模板和内容都留出，cell F1 与全表完全正确率 |
| 23 | 公共设施导览与位置问答 | 合规公共设施图、受权 Open Images 子集，人工标空间方位，回答 SFT | 场地和拍摄者分组盲测，方向 macro-F1 与不可见误报 |
| 24 | 细粒度材质识别 | ABO 带真属性记录与自建光照变化样例，属性 SFT 与违反视觉证据的偏好候选 | 未见材质族或同款组，macro-F1、跨光照一致率 |
| 25 | 信息不足时可解释拒答 | VizWiz 不可回答和自建裁切反事实样本，先 SFT 再做拒答成对对照 | 跨来源盲测，固定回答覆盖率下错误率与误拒率 |

### SDXL 1.0

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 26 ★ | 电商商品棚拍 | ABO 单主体目录图，UNet Attention LoRA SFT，商品与背景风格配对隔离 | 未见商品类别组合上的 CLIP Score、匹配真实图分布 FID，人工棚拍适用率 |
| 27 | 家具与地面场景 | ABO 家具和许可的合成房间，构图条件描述 SFT | 留出家具家族和房间模板，CLIP Score、布局约束通过率 |
| 28 | 与提示一致的配色 | 自建已知色卡和许可 COCO 单物体对，SFT 后独立复现 DDPO 颜色奖励 | 未见物体×颜色组合，独立检测器和人工的颜色绑定准确率，CLIP Score/FID |
| 29 | 同提示下减少构图瑕疵 | 按原图许可筛选 Pick-a-Pic v2，或自标的固定底座成对样本，Diffusion-DPO LoRA | 未见提示簇与独立盲评员成对胜率，图像失效率和 CLIP Score |

### Z-Image-Turbo

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 30 ★ | 目录商品纯背景或透明底风格 | ABO 合规单体商品图，融合原 Turbo adapter 后训练 Differential LoRA | 未见 SKU 家族，背景合格率、主体完整率、CLIP Score/FID |
| 31 | 家居广告留白与主体布局 | ABO/PBR 授权资产自渲染，Differential LoRA SFT，图面保留区域结构化标注 | 未见布局模板和商品家族，主体框和留白合格率 |
| 32 | 指定颜色绑定到指定物体 | 许可 COCO 与合成多物体对，先走原 SFT，再复现 Flow-GRPO 的实际策略更新 | 未见组合与 GenEval2 属性子集，颜色绑定准确率，CLIP Score/FID |
| 33 | 馆藏陶瓷的器型和釉色生成 | The Met Open Access 中有 CC0 标记的陶瓷图与馆藏结构化描述，Differential LoRA SFT | 作品家族与创作者分组留出，器型和釉色盲评通过率、CLIP Score |

### HY-Motion 1.0

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 34 ★ | 步行、急停和原地转向 | ACCAD/CMU 合格公开动作及自主文本，原 128 帧条件和 201 维目标的 Flow Matching LoRA | 按表演者及原记录留出，方向误差、足滑、R@1、MotionFD |
| 35 | 弯腰、拾取和放置的动作顺序 | 许可合格的 HumanML3D/AMASS 分层样本，重定向后保持原目标，SFT 监督完整动作 | 未见表演者与原始序列上的顺序执行比例、R@1、MotionFD |
| 36 | 长动作段的方向与脚接触 | ACCAD/CMU 可复用的长段与停转段，先 SFT，再验证原 MotionRFT 配方 | 独立原序列盲测，脚接触召回、足滑速度和方向错误率，与 R@1 同时报 |
| 37 | 独舞动作与连接步 | AIST++ 的有权动作参数、自撰写动作文字，转换为原 201 维后做 SFT 和动作偏好对照 | 按舞者和编舞均留出，动作完成率、过渡关节加速度、MotionFD |

### HunyuanVideo 1.5

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 38 ★ | 桌面单品转台短视频 | ABO 多视角真实转台或自渲染序列，生成原缓存，将 480p T2V Transformer LoRA 用于物体一致性 | SKU 家族留出，Video CLIP、VBench subject consistency、闪烁率 |
| 39 | 室内场景稳定运镜 | TartanAir 和授权室内自渲染，使用真实轨迹信息标注定镜、推拉和横移，按原缓存配方 SFT | 未见室内布局和轨迹族，人工运镜类型识别、VBench motion smoothness |
| 40 | 自然场景稳定运动 | 只取逐条许可明确的 USGS/Commons 自然视频，去掉重合源片段，T2V LoRA SFT | 留出地点、原片和拍摄者，Video CLIP、背景一致性、运动平滑 |
| 41 | 文本动作与时间对齐 | 自渲染与 Netflix Open Content 可许可训练片段建立动作对照，SFT 后复现 SAGE-GRPO，保留原 1 actor、1 VAE 与 4×4 的首步参考 | 整片来源与动作组合留出，VBench 文本相关维度、Video CLIP、独立盲评动作完成率 |

### TRELLIS.2

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 42 ★ | 单张商品图重建硬表面 | ABO 可匹配的带网格对象与自渲染图，冻结外部管线，只训练原 1.3B Shape Flow LoRA | 按原 mesh 和商品家族隔离，Chamfer-L1、F-score@1%、网格导出失败率 |
| 43 | 复杂家具细长结构重建 | ABO 与 Poly Haven 合规网格，细腿、靠背分层采样，使用原生成缓存和 Shape Flow LoRA | 未见家具家族，Chamfer-L1、F-score@1% 和预定义细结构召回 |
| 44 | 遮挡下的物体几何补全 | 使用 ABO/Poly Haven 网格自渲染许可遮挡图，训练目标仍为原完整网格 | 未见物体资产与遮挡模板，完整网格 F-score@1%、被遮部分召回 |

### Stable Fast 3D

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 45 ★ | 商品 PBR 材质还原 | ABO/Poly Haven 有 PBR 源值的资产和固定视角 RGBA，训练原 36 个材质头张量 | 未见材质族和物体的固定视角 LPIPS、albedo/roughness/metallic MAE |
| 46 | 不同强度光照下的固有材质 | Poly Haven CC0 HDRI 和材质，同一资产不同照明配对，原材质头加一致性损失对照 | 留出材质和 HDRI，材质 MAE、跨照明差异与 LPIPS |
| 47 | 混合金属和非金属的材质区域 | ABO 带可靠 PBR 标签加自产区域 mask，冻结几何，只训练原三个输出头 | 未见金属组合资产，分区材质 MAE、区域边界和固定视角 LPIPS |

### VGGT-1B

| 编号 | 定制能力 | 数据与后训练 | 独立验收 |
|---|---|---|---|
| 48 ★ | 室内 RGB 双帧相对位姿 | TUM RGB-D 有真值序列，仅输入 RGB；冻 aggregator 训练 camera head | TUM 完整真实序列及场景留出，AUC@30 主指标、AUC@5/15 与失败率 |
| 49 | 商品环拍的大视差位姿 | ABO 原 mesh 自渲染已知外参图，不把自然转台帧默认为标定真值，只训练 camera head | 未见 SKU 家族、资产和角距区间 AUC@5/15/30，另报高角距失败率 |
| 50 | 暗光与运动模糊下位姿 | TartanAir/TUM 可验证位姿样本和受控合成退化，camera head 鲁棒训练 | 留出真实采集序列和未见退化条件，AUC@5/15/30、正常场景退化和失败率 |

## 数据与复现入口

| 方向 | 第一手入口与首个可做的核验 |
|---|---|
| 街景、文档、图表 | TextOCR 与 TextVQA 共享图像标识，TextCaps 也需统一图级去重。DocVQA、ChartQA 和 ChartQAPro 按底层文档或图表来源去重，逐项核图像与标注用途。世界银行自带第三方数据时再核单项许可。 [TextVQA](https://textvqa.org/dataset/) · [TextOCR](https://textvqa.org/textocr/dataset/) · [TextCaps](https://textvqa.org/textcaps/download/) · [DocVQA](https://site.docvqa.org/datasets/docvqa) · [ChartQA](https://github.com/vis-nlp/ChartQA) · [ChartQAPro](https://aclanthology.org/2025.findings-acl.978/) · [世界银行数据许可](https://datacatalog.worldbank.org/public-licenses) |
| 无障碍与视觉推理 | 使用新版 VizWiz，不尝试恢复已经遮蔽的信息。GQA 用官方 scorer，同时记录公开底座在这些基准上的可能预训练重合，因此新增不同来源的自建盲测。ScienceQA 和 Flickr30k 先核素材与标注权限。 [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/) · [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/) · [GQA](https://cs.stanford.edu/~dorarad/gqa/evaluate.html) · [ScienceQA](https://scienceqa.github.io/) · [Flickr30k Entities](https://github.com/BryanPlummer/flickr30k_entities) · [MathVista](https://mathvista.github.io/) |
| 图像、商品与 3D | ABO 当前官方入口标 CC BY 4.0，保留来源及署名，检查商品图与可匹配网格子集。Poly Haven 资产为 CC0，可用于自渲染真值。The Met 只用馆方标记为 Open Access 的作品。先跑通真实 TRELLIS.2 的 O-Voxel/latent 数据生产，SF3D 用渲染素材里的已知材质或原贴图生成真值。 [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [Poly Haven](https://polyhaven.com/license) · [The Met Open Access](https://www.metmuseum.org/policies/frequently-asked-questions-image-and-data-resources) · [TRELLIS.2 数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) |
| 动作与视频 | 先用 CMU 的原始动作或已有已核验的 ACCAD 源。HumanML3D 从 AMASS 等上游来源衍生，继承的限制逐项核实。AIST++ 只考虑动作参数，不取评测用音乐或视频素材，另核参数转换所需的模型许可。几何合格的动作还要通过原生文本编码。Netflix Open Content 标 CC BY 4.0，按作品和镜头隔离，先取短片段验证解码与色彩转换。 [CMU](https://mocap.cs.cmu.edu/faqs.php) · [ACCAD](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) · [HumanML3D](https://github.com/d3ac/HumanML3D) · [AMASS](https://amass.is.tue.mpg.de/license.html) · [AIST++ 下载](https://google.github.io/aistplusplus_dataset/download.html) · [USGS 版权与署名](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits) · [Wikimedia 图片许可](https://commons.wikimedia.org/wiki/Commons:Licensing) · [Netflix Open Content](https://opencontent.netflix.com/) · [VBench](https://github.com/Vchitect/VBench) |
| 摄像机 | 以原始采集序列或完整合成轨迹作划分。用真值文件绑定时间戳、坐标系、相机内参和外参约定。商业交付如需切换 VGGT-1B-Commercial，单立底座、A100 参考和质量报告。 [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [TartanAir](https://tartanair.org/) · [CO3D 条款](https://github.com/facebookresearch/co3d/blob/main/LICENSE) · [VGGT 权重许可说明](https://github.com/facebookresearch/vggt#license) |

专项交付前，核对权重来源和发布通知。Gemma、SDXL、HY-Motion、HunyuanVideo、SF3D 和 VGGT 的具体底座许可影响可交付的适配器、合并权重或接收环境。VGGT 原权重限非商用，商业版本须重新适配验证。条款证据写进每个产物的 model card。 [Gemma 条款](https://ai.google.dev/gemma/terms) · [SDXL 权重卡](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) · [HY-Motion 许可](https://github.com/Tencent-Hunyuan/HY-Motion-1.0/blob/master/License.txt) · [HunyuanVideo 许可](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE) · [SF3D 许可](https://github.com/Stability-AI/stable-fast-3d/blob/main/LICENSE.md) · [VGGT 发布说明](https://github.com/facebookresearch/vggt)

### 论文复现的实验对照

| 方法 | 计划编号 | 首个验证与正式对照 |
|---|---|---|
| LoRA SFT | 各底座首项及多数候选 | 沿用资格已验证的冻结边界和更新器。比较原底座、同配方通用 SFT、同预算领域 SFT，固定数据顺序与首步证据。 [LoRA 原论文](https://arxiv.org/abs/2106.09685) |
| 视觉问答偏好优化 | 09、16、24、25 | 先用 SFT 获得可回答模型，再用同图、同提示的事实差异对。冻结 reference，核验 chosen/rejected log-prob、梯度和更新，再与同预算 SFT 对比；优胜标准用盲测事实错误率、覆盖率和胜率。 [DPO 原论文](https://arxiv.org/abs/2305.18290) |
| DDPO 与 Diffusion-DPO | 28、29 | 28 沿用 4 条 rollout × 4 去噪步和原颜色奖励。29 单立离线偏好配方，用同提示、固定噪声的带偏好图对，独立盲评员计胜率。两项均复核冻结参考及更新是否发生。 [DDPO 原论文](https://arxiv.org/abs/2305.13301) · [Diffusion-DPO 原论文](https://arxiv.org/abs/2311.12908) · [Pick-a-Pic](https://github.com/yuvalkirstain/PickScore) |
| Flow-GRPO | 32 | 锁定原 Turbo 融合权重、每提示 4 条 rollout × 8 采样步。用原奖励验证分项及策略更新，再用独立色彩绑定评分确认可迁移。 [Flow-GRPO 官方实现](https://github.com/yifan123/flow_grpo) · [GenEval2](https://github.com/facebookresearch/GenEval2) |
| MotionRFT 与动作偏好 | 36、37 | 36 先复现 MotionRFT 的 4×4、文本/动作各 0.5。37 等原生输出和文本缓存可用后，再单立偏好配方，参考 MoDiPO 的方法，并验证向现有 Flow Matching 入口的转换。两项都用独立运动学指标检验收益。 [MotionRFT 官方实现](https://github.com/Xiaofeng-Tan/MotionRFT) · [MoDiPO 论文](https://arxiv.org/abs/2405.03803) |
| SAGE-GRPO | 41 | 在原 HunyuanVideo 480p SFT 后核原本地奖励、old/new log-prob、ratio、advantage 及策略更新。论文的 VideoAlign 与用户原本地复合奖励分别记录，VBench 与人工盲评作为验收。 [SAGE-GRPO 官方实现](https://github.com/Tencent-Hunyuan/SAGE-GRPO) |

## 推进与交付

| 批次 | 范围 | 当批验收 |
|---|---|---|
| 第一批 | 12 个带 ★ 的编号 | 每个底座交付至少 1 个完成真实参数更新、独立盲测达标且通过加载复验的领域模型。若首选方向没有复核收益，保留结果并推进同底座下一候选，直至该底座具备达标模型。 |
| 第二批 | 其余 24 个常规专项 | 复用已合格的数据处理和评测入口，按实测卡时预算滚动派发 |
| 第三批 | 09、12、14、15、18、25、28、29、32、35、36、37、40、41 | 在许可、额外算子/奖励、原始资料、独立评测器齐备后再准入，重点交付偏好与强化学习有独立收益的版本 |

首轮并行准备 01、06、26、42、48 的数据准入，分别验证图文 QA、不可回答标注、商品图、三维 latent 和相机真值五条入口。其余首批同步查找可直接复用的已有冻结数据。34 依赖实际文本缓存，38 依赖视频缓存和真值身份，45 依赖可逐值复算的材质标签。训练资源按 `每次实际卡数 × 实际耗时` 记卡时，pilot 用每步和每样本实测外推长训预算，预留跨平台复测和盲评分的独立预算。

1. **专门智能体维护数据和方法。** 每项按数据来源、许可、训练 split、留出 split、最小可执行配方、预计卡时、评测器和实际运行记录推进。每次交接至少产生一项可复核产物，如数据清单、独立切分、25 步复现结果或失败归档。训练数据优先使用现有合格材料。准备新数据时只拉最小可用样本到项目存储，先过格式和许可，再申请全量处理。
2. **调度与原项目共用资源。** 在原训练分区滚动运行，保持原三模型推理和算子调优的分配。调度以实际 Slurm 所持节点、每卡设备使用率和原生入口的最小可运行拓扑为依据。VLM 的正式跨平台性能参考仍用 8 卡 BF16、每卡 batch 1、累积 4。其他规模或训练范围另发参考。空闲卡可优先做合格数据缓存、复评或回归，资源报表单列等待、有效训练、评测与低优先级补位卡时。
3. **先定盲测门槛，再看结果。** 冻结每行的主指标、正确方向、最小有用差异和样本数。以原底座作配对比较，按原始图、视频、人物或资产分组计算 95% 置信区间。建议初筛用准确率类至少 +3 个百分点，误差类至少下降 10%，低样本或高基线任务由 dev 预先另定。核心效果退化仍套原阶段门槛。所有坏输出、解码失败、无可计分与弃答留在报告；只有训练后质量通过且产物可独立加载才记交付。
4. **归档每个可加载的模型。** 产物包含 adapter 或头权重，允许时附合并权重、原底座与框架 SHA、模型卡与使用限制、恢复检查点、首步与 25 步结果、固定盲测用例、逐样本评分、训练前后和 A100 对照、三轮计时中位数及最短运行命令。独立进程从交付包加载，用预留样例复验。模型权重和原始数据只放项目产物存储，Git 记源码、清单和带摘要的结果。

进度汇总同时报首批已交付底座数（分母 12）和累计已交付领域模型数（分母 50）。按底座列出数据准入、可提交、实际更新、盲测通过和可加载产物。另报失败分母、有效训练卡时、补位卡时、空闲卡时和下一批可启动作业。
