# 12 个底座的领域模型接续执行单

先交付每个底座的一个可复验领域模型，再向 50 个扩展。按证据缺口排队。已有检查点直接评测和恢复，已核数据推进实际更新，缺原生样本的线路先做数据和缓存。所有训练保留原框架入口及训练范围。领域验收和 17 项 A100 对照各自计数。 [50 项总路线](specialized-posttraining-50-v1.zh-CN.md)

| 底座 | 首发能力和已有证据 | 下一项执行与通过条件 | 备选方向 |
|---|---|---|---|
| Gemma-3-4B-IT | USGS 地震图表读数。CP25 已在新进程加载，96 题为 52/96 对 40/96，提升 12.5 个百分点。 | 沿原 512 序列 LoRA 查第 21 步恢复差异，跑 TextVQA 与 COCO CIDEr 保持性，核交付包。96 题已经揭盲，后续配方只用新事件组判断。 | VizWiz 低质量图片问答和不可回答检测，按来源图分组，主报 VQA accuracy 和不可回答 AP。 |
| MiniCPM-V-4.6 | Open Food Facts 包装规格和不可读拒答。实际更新 2 步，两臂 14 题预测已封存。 | 按原数据权限复用已封存预测和本机评分器，报数值与单位联合正确数 `/7`、不可读误答数 `/7` 和七组商品配对结果。小批用于决定是否扩未见商品家族。 | 用自建截图做表格单元格提取，按页面模板留出，主报 cell F1。 |
| InternVL3.5-8B-HF | GQA 图中物体关系。已有八卡 CP25 与 96 题两臂预测。 | 按原权限对封存预测评分，保留真值隔离。CP20 恢复偏差另走八 rank Swift 采样和 Arrow 行映射定位。冻结门槛为 ≥3 个百分点且按图像配对的 95% 区间下界大于零。 | 程序渲染加权利明确的科学图题，考查图示关系与证据区域，按概念和图源留出。 |
| HunyuanVideo 1.5 | ABO 桌面商品转台。Q/V LoRA + Muon 已实际更新 25 步，216 个 LoRA 张量对上，原重载步骤设备句柄检查失败。 | 定位 PEFT 导入链的首次 KFD 打开，并在正确放置下新进程加载、正常退出和独立清场。使用原 3 个留出物体做成对生成，验主体一致性、闪烁与 Video CLIP。 | 只改变运动方向的同商品短轨道视频，按资产家族留出，验镜头方向和主体一致性。 |
| VGGT-1B | TUM RGB-D 室内双帧位姿。冻结 aggregator 的相机头 CP25 已重载。 | 接续原目标准入，用同一 224px 双帧入口，对底座和 CP25 各计原 20 槽，保留 1 槽几何失败，主报 AUC@30，附 AUC@5/15。 | 用 TartanAir 有明确位姿的独立轨迹训练相机头，按场景整组切分，真实 TUM 录制做外测。 [TartanAir 发布说明](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-tartanair-simulation) |
| Z-Image-Turbo | 科学图表风格。已有 Style18 CP25 与同一 Turbo 融合底座。 | 复用原生渲染入口和现有盲序代码，核权重、输入、噪声并完成目标端准入。冻结两名评审及盲序后输出 8 个提示 × 2 种子 × 2 臂，主报领域风格与结构同时合格率，附通用控制。 | 官方 ABO 或自渲染的目录单品无干扰背景，验主体完整率、背景合格率、CLIP Score 与 FID。 |
| Qwen3-VL-8B-Instruct | TextVQA/TextOCR 街景读字和定位。41 条已有原生输入验证。 | 对账原八卡请求和节点。使用原 BF16 拓扑，保留视觉冻结，核前两个实际更新中的正学习率参数差，再跑连续 25 步。按原图去重，报官方 TextVQA 与小字、多行、反光分桶。 | 自建文档模板做结构化字段抽取，按发行方模板整组留出，主报字段 micro-F1。 |
| LLaVA-OneVision-1.5-8B-Instruct | 物体清单与低幻觉图像描述。正式方案用有逐图许可和实例标注的 COCO。 | 权重分片核齐期间施工 CLEVR 官方小样本。用公开场景图监督颜色、形状、数量，走原 OneVision eager SFT。先测原生读取，再测真实参数更新。CLEVR 仅用于场景测试与开发，正式效果由真实图片 CHAIR/CIDEr 与固定盲评判断。 | 若 COCO 权利或对象标注配对不足，延续 CC BY 4.0 场景图训练，独立制作未见实拍物体清单作为外测。 [CLEVR 官方数据及许可](https://cs.stanford.edu/people/jcjohns/clevr/) · [CHAIR 论文](https://aclanthology.org/D18-1437/) |
| SDXL 1.0 | ABO 电商商品棚拍。保留原 Attention LoRA，冻结文本编码器和 VAE。 | 选同一商品目录图与可见描述，对账噪声、timestep 及首步反向梯度差。首步通过进入连续更新。按商品家族和背景模板留出，验棚拍适用率、CLIP Score 和 FID。 | Poly Haven CC0 商品自渲染配色，固定几何和相机，用未见物体×颜色组合验颜色绑定。 |
| TRELLIS.2 | 工具和家居硬表面单图重建。已有完整网格和表面回读候选。 | 用一件同源网格和渲染过原生 O-Voxel、Shape latent 及条件，绑定 1024/2048 上限，再试 Shape Flow LoRA 首步。按资产家族留出，验 Chamfer-L1、F-score@1% 和导出失败。 | ABO 无歧义网格配对不足时用 Poly Haven CC0 新家族。 [原生数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) |
| Stable Fast 3D | 商品 PBR 材质。已有 `metal_tool_chest` 固定视角及同像素材质回读。 | 绑定实际冻结特征和 36 个材质张量，在有前景真值像素的输入上核参数变化。用不同物体家族和物理材质验固定视角 LPIPS 和 albedo、roughness、metallic 三项 MAE。 | Poly Haven 不透明木与塑料类别，按光照、材质家族和物体同时留出。 |
| HY-Motion 1.0 | 行走、急停、转向。ACCAD 已有源骨架运动学证据。 | 用一条真实源片段打通骨架重定向、128 帧 × 201 维目标、文本缓存和足地接触，再实跑 Flow Matching LoRA。按表演者与录制隔离，验 R@1、MotionFD、足滑和方向误差。 | ACCAD 单条映射仍有歧义时，用相应 CMU 原始动作与明确转换验证绑定，再按同配方重新准入。 [ACCAD](https://accad.osu.edu/research/motion-lab/mocap-system-and-data) · [CMU](https://mocap.cs.cmu.edu/) |

## 并行派发

- **最短评分路径**。先处理 MiniCPM 与 InternVL 已封存预测，按各自具体权限核对；Gemma 复用已有盲测完成恢复与保持性；VGGT、Z-Image、HunyuanVideo 分别推进位姿、配对渲染和生成。评分与清场任务单独登记。
- **首批五个训练储备候选**。按缺口最少排序，当前先准备 Qwen 的原八卡接续、MiniCPM 新商品组、InternVL GQA56 恢复定位后的训练、SF3D 固定材质首步、LLaVA 场景清单首步。各项在精确输入、权重、启动命令、正学习率更新和实际放置核齐后才占一个就绪槽。SDXL、TRELLIS、HY-Motion 并行补数据，可先就绪先替补。
- **数据复用**。ABO/Poly Haven 资产先按原网格和家族确定划分，再派生棚拍图、转台、材质真值和几何缓存。派生物保留原资产身份及相机设置。各模型训练和留出在原资产组上隔离。 [ABO 官方入口](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [Poly Haven CC0](https://polyhaven.com/license)
- **早停和交付**。先取得正学习率参数变化，再完成连续更新和恢复。用开发集决定训练长度，正式盲测前固定主指标、最小改善、配对分组、失败分母、通用能力保持及同预算对照。同底座首选在冻结盲测无收益时归档结果，转备选。交付须有原底座身份、可加载权重或适配器、新进程复验和启动命令。

[前序执行单及证据索引](specialized-posttraining-12-accelerated-execution-20261127.zh-CN.md) · [InternVL 八 rank 原失败](../../results/bw1000/2026-11-27/specialized-internvl-4638-cpu-sampler-gloo-filestore-no-permutation.json) · [Gemma 与 HunyuanVideo 实测补记](../../results/bw1000/2026-11-27/training-cluster-daily-2055-domain-readback-and-fleet.zh-CN.md)

## 本轮实际产物

- LLaVA 的 CLEVR 小批已落盘。独立解码核验 96 张原图、零内容哈希重复。64 条训练行含场景图监督，16 条开发行和 16 条校准行不带答案。只读了官方 ZIP 的约 69.5 MB 范围。复用原 `.378` 的 OneVision 模板、预处理器和元权重路径，先核新样本的原生监督 mask。第 1 权重分片上次独审核对后仍缺约 553 MB，待重新核对完整分片与初始化权重。真实图片盲测另走 COCO 外测。 [样本与独立解码回执](../../results/bw1000/2026-11-27/specialized-llava-clevr-official96-stage-and-independent-decode.json) · [原预处理证据](../../results/bw1000/2026-11-19/training-llava-original378-native64-meta504-and-A100-input-match.json) · [第 1 分片前缀独审](../../results/bw1000/2026-11-22/training-llava-original-shard01-target-prefix-independently-qualified.json)
- InternVL 的共享结果目录 FileStore 试验中 Gloo 返回 0/8。在同镜像核实用户拥有的可写 sticky tmpfs 后，节点本地 FileStore 试验 `1103606.4642` 中 Gloo 返回 **8/8**，耗时 39–65 ms，全部正常销毁。随后诊断误把采样器内部持有的每 rank 8 槽当作全局 64 槽，目标源码已确认这一差异。改正诊断期望后继续验证实际排列，再验 Arrow 映射。 [原初始化终态](../../results/bw1000/2026-11-27/specialized-internvl-4639-eight-ranks-stalled-in-gloo-init.json) · [同镜像挂载核验](../../results/bw1000/2026-11-27/specialized-internvl-4640-container-shm-owner-witness.json) · [本地 FileStore 终态](../../results/bw1000/2026-11-27/specialized-internvl-4642-gloo-eight-returned-swift-guard.json) · [已部署 Swift 源码合同](../../results/bw1000/2026-11-27/specialized-internvl-4642-deployed-swift-total-samples-contract.json)
- 修正后的 `1103606.4643` 在目标端实际完成八 rank 原生采样，各 rank 取 8 槽，合计恰好覆盖 64 个逻辑槽，8/8 正常退出。下一项是用原生加载器对真实 Arrow 行及来源身份，训练作业仍待该环节及真实输入顺序核验。 [原生采样终态](../../results/bw1000/2026-11-27/specialized-internvl-4643-native-eight-rank-logical-sampler-pass.json)
- Z-Image 复核了现有 8 个提示、16 对和 32 个计划生成槽，以及原 CP25 与基础状态摘要。沿原渲染包推进，仍需两名已确认的独立评审、生成前盲序和目标端准入。 [原生执行交接](../../code/bw1000/training-zimage-style18-native-heldout-render-v1/HANDOFF.md)
