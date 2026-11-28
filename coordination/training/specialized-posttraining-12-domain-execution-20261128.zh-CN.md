# 十二个底座的领域模型推进单

首批按每个底座一个可独立加载的领域适配器或头权重推进。先验收已取得的检查点，再接续训练；同时为尚未取得原生更新的底座准备最小可运行样本。下列留出数量是首轮执行预算，正式独立盲测按来源分组，在训练和查看候选结果前冻结。已有正式冻结的配方和阈值按原文件执行。

## 加速排期与每个底座的下一件产物

三组并行准备。目标端能使用现有检查点时，先完成独立评分或载入；还缺首步证据时，先交一条源文件可追溯、原生入口可读的训练样本。训练与评分使用统一框架中已合格的模型入口，样本准备在 CPU 单独进行。训练优先使用已核准的原拓扑，每次交接只替换实际需要的保底资源。

| 优先组 | 底座 | 最近可交付的产物和后续进入条件 |
|---|---|---|
| A | Gemma | 封存 USGS CP25 领域盲测 52/96 对 40/96。补齐 TextVQA 2,275 张未核图片并排除训练重叠，接同题保持性和 COCO CIDEr；另定位从 CP20 恢复的第 21 步差异。已有候选权重优先兑现。 |
| A | InternVL | 复用 GQA CP25、504 个逐值核验的 LoRA 张量及已冻结的 96 图。先对账原载体是否送达，再做 192 个配对预测。查询和判断各 48 题，按图 bootstrap。原传输结果未知时保留原请求。 |
| A | MiniCPM | 优先独立评分现有七个商品组的 28 条封存输出，并保留可读与拒答分母。原端评分需要同端已有、允许使用的答案与冻结评分器。并行选 Open Food Facts 新商品组，逐图核共享条款及跨家族隔离，扩足后再做连续更新。 |
| A | LLaVA | 对目标端已核的 64 个 CLEVR 样本走原八卡、eager 的 Swift 入口。首批取得 64 次前向、第二步正 LR 下的参数变化与 CP2；其后补真实图片的物体存在判断与自由描述，分别评分 POPE 和 CHAIR。 |
| B | SF3D | 先在已验 CC0 网格上真实渲染物体级恒定 PBR，复核 RGBA、线性 albedo、roughness 和 metallic，再把新 cohort 接入冻结特征和原 36 张量材质头。空间变化的贴图继续用于逐点评测研究。 |
| B | SDXL | 第二文本编码器继续原分片交接，整件校验后产两路官方文本条件，绑定已过 VAE 的工具柜和漆凳，再用原 UNet LoRA 跑首个有效更新。商品颜色与背景按资产家族分层。 |
| B | HunyuanVideo | 继续取得合法单卡窗口的新进程 CP25 复载，验证 216 个 LoRA 张量和设备释放，通过后跑已固定的三商品配对生成。新训练素材复用 CC0 商品转台及同物体的原生 480p 文本和 VAE 缓存。 |
| B | Z-Image-Turbo | 先固定独立双评审和盲序，再出 Style18 CP25 的 16 对图。评审条件持续不足时，单独准备 Met Open Access 明确标注的陶瓷，按器形与馆藏对象留出。 [大都会艺术博物馆开放馆藏](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access) |
| C | Qwen3-VL | 先查原八卡请求，再用 41 条已过原生输入的街景 OCR 监督取得有效更新。ST-VQA 扩小字与多行，先逐图核许可并按原图划分，TextVQA/TextOCR 同图只进入一个集合。 [ST-VQA 发布方](https://rrc.cvc.uab.es/?ch=11) · [原论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Biten_Scene_Text_Visual_Question_Answering_ICCV_2019_paper.html) |
| C | VGGT | 原 TUM 新进程对照按 20 槽计分，保留一个几何失败。TartanAir v2 备训练时先选一条室内左前相机轨迹，用 NED 位姿、640 像素原图内参和同帧深度核双向重投影，记录原配方 224 像素预处理后的内参。按录制和场景隔离，TUM 留作真实外测。 [TartanAir 相机数据](https://tartanair.org/modalities.html#raw-camera-pose) · [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) |
| C | TRELLIS.2 | 用训练组工具柜生成 O-Voxel 和 Shape latent，原 Shape Flow 数据集实读稀疏坐标、形状特征和条件视图，核 1024/2048 上限。录音机或电水壶可替补；保留独立网格组做 Chamfer-L1 与 F-score@1%。 [微软原生数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) |
| C | HY-Motion | ACCAD 按发布者逐条配对标定与同步记录。备选 CMU Subject 91 用于转向，Subject 128 的 Run Stop Run 用于急停后再启动，两者按表演者留出。先核各自骨架、帧率、时序覆盖及原 `<40 mm` 几何门，再生成原生的 128×201 训练张量。 [ACCAD](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) · [CMU 转向](https://mocap.cs.cmu.edu/search.php?subjectnumber=91) · [CMU 急停](https://mocap.cs.cmu.edu/search.php?subjectnumber=128) |

每个底座独立记录数据、训练、恢复、领域盲测和可加载权重五项证据。已形成真实权重的候选优先用现有检查点补证。单一底座的额外变体等首批全覆盖后再投入。共用商品原资产清单、许可、哈希和家族划分，SDXL、HunyuanVideo、TRELLIS.2 与 SF3D 分别保存自身原生数据回执。替补启用前冻结自己的验证集、门槛和原底座对照，保留首选失败结果。

- SF3D 已生成并独立读回一张固定视角恒定 PBR 的 256×256 RGBA。20,600 个可见像素、97 个标量取点及三个原生标签形状均已验证，原生特征和训练待验。 [样本回执](../../results/bw1000/2026-11-28/specialized-sf3d-cc0-constant-pbr-cpu-render-and-independent-readback.json)
- LLaVA 新控制步骤 `1103606.4978` 在旧保底终态证据的原始等待窗口内失败，训练 GPU 未派发。独立 `1103606.4979` 已核对应八卡、KFD/FD 及训练端口。下一次在当前机位独占与原证据均复核后，从已完成的保底终态接管。 [实际终态](../../results/bw1000/2026-11-28/specialized-llava-4978-release-proof-timeout-and-4979-clean.json)
- LLaVA 后续八卡 `1103606.4983` 正常结束，两步累计 64 次有效前向和 16 条 rank 优化器返回。第一步学习率为 0，第二步学习率为 `0.0001`、参数差 L2 为 `0.426978`。CP2 的 17 个文件已校验，含优化器、调度器和八个 rank 的 RNG。`1103606.4984` 复核了同 boot、八 PCI、设备句柄及端口清场。下一项在统一 Swift 入口从 CP2 恢复，核后续数据顺序和数值，再推进 25 步与留出集。 [实际更新与 CP2](../../results/bw1000/2026-11-28/specialized-llava-4983-eight-rank-first2-cp2-and-4984-clean.json)
- SF3D 新样本已在本机用原解析逻辑读取 RGBA 与三项 FP32 标签，错误顺序和清单被拒绝。后端至少需要两个样本，第二件用已核 CC0 漆木凳重渲染恒定非金属材质，再验证真实冻结编码器和 36 个原生可训练张量。 [本机读取边界](../../results/bw1000/2026-11-28/specialized-sf3d-constant-pbr-one-view-native-parser-local.json)

| 底座 | 首交付能力和数据 | 下一次实际执行 | 领域验收 |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | 小字、多行街景 OCR。复用已过原生处理的 26 条 TextVQA 和 15 条 TextOCR；按原图去重，先保留可见答案，再补定位目标。 | 先查原唯一八卡请求，使用 `qwen3_vl`、BF16、SDPA、原 LoRA 冻结范围及 4096 上限，核首个正学习率和参数差，接 25 步。 | 独立 TextVQA 评分，按原图配对，报告小字、多行、反光和未作答分母。 [TextVQA](https://textvqa.org/dataset/) · [TextOCR](https://textvqa.org/textocr/dataset/) |
| Gemma-3-4B-IT | USGS 地震图表读数。已核验 CP25 在 96 题中答对 52 题，原底座 40 题，按事件分组 95% 区间为 +3.125 至 +20.833 个百分点。 | 复用原 CP25 和新进程加载结果。定位 CP20 恢复后的第 21 步数值分歧，继续补独立 TextVQA 与 COCO CIDEr 保持性；已有的 96 题封存。 | 后续版本另取地震事件，按事件家族留出。同一贪心设置核最终读数及单位。 [独立结果和恢复差异](../../results/bw1000/2026-11-27/specialized-gemma-usgs-cp25-release-audit.json) · [USGS 公开地震目录](https://earthquake.usgs.gov/fdsnws/event/1/) |
| InternVL3.5-8B-HF | 物体关系查询、关系真假判断。GQA 56 个独立训练图；新盲测按图、来源和近重复排除训练、工程和旧评测集合。 | 八卡已完成到 CP25，新进程已逐值复核 504 个适配器张量。准备以本次 CP25 和原底座评新盲测，先复核原生模板逐槽输入及适配器身份，再启两臂生成。 | 96 张独立图，关系查询和真假判断各 48 题。归一化整答匹配，缺失和达到生成上限计错。按图做配对 bootstrap，要求至少 +3 个百分点且 95% 下界大于零。 [GQA 官方资料](https://cs.stanford.edu/people/dorarad/gqa/download.html) |
| LLaVA-OneVision-1.5-8B-Instruct | 减少物体幻觉。CLEVR 64 个训练行已过原生监督检查，16 个开发和 16 个校准按图隔离，再用逐图许可的真实图片构造 grounded captions。 | 四块权重和 14 个元数据文件已组装为同一目标目录，独立核对 698 项索引。核实 Swift 原生 eager 和 LoRA 实体加载后，做首步，检查 assistant mask、梯度和参数差。 | CLEVR 记存在错误。真实图像以 CHAIR 对物体事实计分，报 CIDEr、漏报和生成失败。 [完整模型目录](../../results/bw1000/2026-11-28/specialized-llava-full-official-model18-and-independent-698-index.json) · [CLEVR 官方数据](https://cs.stanford.edu/people/jcjohns/clevr/) · [CHAIR](https://aclanthology.org/D18-1437/) |
| MiniCPM-V-4.6 | 包装规格、单位和不可读拒答。使用已合格 Open Food Facts 小批，按商品家族留出，先增加实际可读和同类遮挡两类。 | 复用已实跑两步和独立复载的 CP2。优先处理原 14 对封存输出；并行补新家族，保持冻结视觉和对齐模块及 5 步 warmup。 | 规格数字与单位联合准确率，另报不可读误答和有答案误拒。原 14 对属于小样。 [Open Food Facts 数据条件](https://world.openfoodfacts.org/data) |
| SDXL 1.0 | 电商单品棚拍。先取一件权利核验过的 ABO 商品图或 Poly Haven 渲染，caption 只描述图中可见物体，再扩成按资产家族隔离的小批。 | 用原 256×256、batch1、UNet Attention LoRA rank4/alpha4，冻结文本编码器和 VAE；固定噪声、timestep，首圈只定位反向差异并验证有效更新。 | 同提示和同种子做单品完整、背景合格双指标；大样本用 CLIP Score/FID，样本数一致。 [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) · [Poly Haven CC0](https://polyhaven.com/license) |
| Z-Image-Turbo | 科学插图风格。优先复用 Style18 CP25，维持预置 Turbo adapter 融合后底座和原 Differential LoRA。 | 锁两名独立评审与盲序；4 个领域、4 个通用提示各 2 个种子，共 16 对，核原生 VAE 和生成入口，再运行。 | 领域风格和结构同时合格率，通用保持与生成失败单列。若无可复核领域收益，走公共领域馆藏陶瓷的替补。 [既有生成包和评审前提](../../results/bw1000/2026-11-27/specialized-zimage-style18-independent-blinding-and-native-render-package-local.json) · [The Met Open Access](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access) |
| HY-Motion 1.0 | 步行、急停和原地转向。ACCAD 原动作按表演者/记录分组；若同配方原生编码缺可用数据，用不同表演者、动作同类的 CMU 原记录。 | 先生成一条真实 128 帧、完整 201 维缓存，核旋转、接触、root 速度和文本条件；再以原 qkv/proj/fc1/fc2 LoRA rank4/alpha8 做有效更新。 | 除方向误差、足滑，保留 R@1、MotionFD。足滑与 R@1 分别报告，MotionRFT 用独立用例。 [ACCAD](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) · [CMU](https://mocap.cs.cmu.edu/faqs.php) |
| HunyuanVideo 1.5 | 480p 桌面商品转台。ABO 已有一组原生训练缓存和三组独立商品共 27 参考帧。 | 原 Q/V LoRA 和 Muon 已完成 25 次更新及独立后置清理。新进程单卡读回在设备保护处失败；从当前模型挂载及已确认的 provider 复核加载边界，然后运行底座和适配器配对生成，同时补不同家族训练缓存。 | 原冻结三件商品做预筛，分商品记主体一致性、闪烁、Video CLIP 和生成失败；扩足独立样本后验收。 [原 25 步和清理](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-v3-4560-25-native-updates-and-4561-independent-post.json) · [单卡读回终态](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-4567-native-adapter-readback-terminal-device-guard-fail.json) · [VBench](https://github.com/Vchitect/VBench) |
| TRELLIS.2 | 单图硬表面重建。已核 Poly Haven 工具柜，并加入漆木凳和墙钟的完整 1K glTF 候选；相近凳子资产作为同一来源家族。 | 先用一个完整网格，冻结对象变换与相机，走官方 `dual_grid`、形状编码与条件渲染，实际取可被原生入口消费的稀疏 latent。核 1024/2048 token 上限，再跑 1.3B Shape Flow LoRA rank2/alpha4。 | 按物体家族独立，冻结点采样与坐标策略，用 Chamfer-L1、F-score@1% 和导出失败率判定。 [微软官方数据制备](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) |
| Stable Fast 3D | 工具、家具和室内物件的 PBR 材质。漆木凳已经产出 1K atlas 与 256 个 UV 数值参照，金属度恒 0；训练数据需要与金属工具柜配对覆盖三种属性。两材质墙钟保留为几何备选。 | 用已核工具柜先对一视角绑定 RGB、前景、可见三维查询点及同像素 PBR 真值，过冻结编码器缓存和原 36 张量更新。漆木凳后续走同一可见输入合同。 | 新资产与材质家族的固定视角 LPIPS、三分项 MAE；按金属/非金属分层。 [工具柜](https://polyhaven.com/a/metal_tool_chest) · [漆木凳](https://polyhaven.com/a/painted_wooden_stool) · [墙钟](https://polyhaven.com/a/vintage_telephone_wall_clock) |
| VGGT-1B | 室内 RGB 双帧相对位姿。继续用 TUM 的真值轨迹，224 像素双帧，冻 aggregator。 | 优先做原底座与现有相机头 CP25 的 household 20 槽核验。19 个现有几何合格对进入原生计分，未合格一槽留在分母，完整录制按序列隔离。 | AUC@30 为主，附 AUC@5/15、平移/旋转误差和失败率。将既有开发试验与正式留出区分。 [TUM 官方数据](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) |

## 本轮调度与替补

1. 优先兑现已有检查点。Gemma 补恢复和保持性，InternVL 走本次 CP25 独立盲测，MiniCPM 的封存对照与新商品家族准备并行。HunyuanVideo 修复单卡重载的设备保护失败后完成两臂预筛；VGGT 等原 20 槽载体准入；Z-Image 先冻结两人盲评顺序。
2. 共享素材身份与派生输入分开记。商品 RGB/视频/几何/PBR 真值分别生成并绑定原资产、相机、变换、版本及校验和；先跑一条原生可读样本，再扩到按资产家族分组的数据。若公开源配对失败，使用已获许可或可自渲染的资产。小样只判断输入和训练入口是否可行。
3. 按具备真实权重、数据、入口、正数更新计划和实际放置的任务计入可提交储备。优先推进 Qwen 首步、LLaVA 首步、SDXL 首步、SF3D 首步、MiniCPM 从 CP2 接续，并行准备 HY-Motion 和 TRELLIS.2 原生缓存。五个 VLM 原训练配方保持单节点 8 卡 BF16、每卡 batch 1、梯度累积 4。已有单卡证据的生成或几何检查继续使用相同验证配方，CPU 数据准备独立于整节点训练排期。每项单独记录可提交、排队、运行、准备和待对账。
4. 交付须同时具备实际更新、适用的连续训练及恢复、独立领域盲测、通用保持和新进程加载。三轮同口径 A100 性能仍归原 17 个基础及独立用例验收。直到每个底座交付一个，才将同底座的其他候选计入累计 50 项。

## 本轮数据产物

本机已从官方源取回两件 CC0 glTF 资产和贴图，独立复读全部 10 个文件共 4,600,248 字节，并核对每个文件的发布方校验值与本机 SHA256。漆木凳生成 1,048,576 个 atlas texel 和 256 个独立 UV 取点，粗糙度有变化、金属度均为 0。墙钟有两套材质，进入几何与渲染候选，超出当前单材质 SF3D 入口的适用范围。目前两件均待按完整物体家族切分，没有新增可供原生训练的行。 [下载及派生证据](../../results/bw1000/2026-11-28/specialized-public-cc0-painted-stool-wall-clock-source-and-atlas.json)

工具柜已有 20,600 个可见像素的三角面谱系和三项材质匹配，原本机回执 SHA256 为 `28c8ca46c2f471dc8864a339681c0111c38cbc2eceab0c35294a054b1cceb197`。SF3D 的下一个证据是在原生冻结编码器中使用该视角并配对特征与材质查询点。SDXL 已从同一视角生成一条 256×256 RGB 和可见物体描述，复读确认 20,600 个前景像素完全保留，待原生 VAE、双文本条件及首步。工具柜按训练家族处理，其他工具柜资产排除于该候选的正式盲测。 [原固定视角](../../results/bw1000/2026-11-27/specialized-sf3d-metal-tool-chest-fixed-view-cpu-pair.json) · [SDXL 图像回执](../../results/bw1000/2026-11-28/specialized-sdxl-metal-tool-chest-cc0-first-256-rgb-source.json)

HY-Motion 的 Male1 B09 已做同源 128×201 几何诊断，原时间留出误差为 42.51 mm，高于 40 mm 门槛。预先指定的 B03 原 C3D 实为 42 点，现有 41 点解析入口在标定前拒绝。下一项先核原发布者的实际 marker labels 和关节对应，或者换用有明确成对标定记录的训练组。保留原 40 mm 门槛及按表演者隔离的盲测。

InternVL 接续作业 `1103606.4783` 在第 4 步开始前因生产跳批函数身份核验失败，8 个 rank 均无新增优化器更新，独立后置步骤已确认资源清理，原 CP3 仍可用。目标源码已证实 Swift 在 `train` 内临时替换 Trainer 的跳批函数，匹配分支会通过 `get_train_dataloader(skip_batches=...)` 创建 loader。下一件事先用真实零卡路径核重建 loader 的 epoch、四个被跳过的槽及首个保留槽，再派发修正版。 [终态与清理证据](../../results/bw1000/2026-11-28/specialized-internvl-4783-native-resume-skip-provider-mismatch.json)

### 续训和原生输入验证

- **InternVL.** 零卡步骤 `1103606.4790` 已核 Swift 的两个 loader、首轮跳过及保留的 32 个位置，以及后续完整 704 个位置。修正版 `1103606.4794` 在原八卡、每秩四 worker 配方下正常完成第 4–25 步。独立复核了 704 次前向、176 条 rank 更新记录、三个检查点的 51 个文件和八秩一致的参数。`1103606.4809` 在新 CPU 进程通过 Swift 加载底座及 PEFT 适配器，504 个 LoRA 张量的 43,646,976 个元素与原 CP25 逐值一致。下一项是按来源隔离的新盲测，同配方不间断数值对照也待验。 [零卡独审](../../results/bw1000/2026-11-28/specialized-internvl-4790-native-swift-skip-independent-704-slots.json) · [八卡运行独审](../../results/bw1000/2026-11-28/specialized-internvl-4794-native-cp3-to-cp25-22-updates-and-51-checkpoint-files.json) · [新进程重载](../../results/bw1000/2026-11-28/specialized-internvl-4809-cp25-fresh-native-reload-504-lora-exact.json)
- **SDXL.** 零卡步骤 `1103606.4792` 对工具柜候选成功运行原训练脚本的双分词调用。两路均有 21 个未截断 token，再分别填充到 77；填充值分别为 49407 和 0。当前 Transformers 返回 CLIP 快速实现，已逐元素对照直接 CLIP 调用。发布方 UNet 的校验值与已有 SCNet 证据一致，可复用该 10.27 GB 文件。VAE、两套文本编码器和完整训练输入仍分项闭合。 [双分词验收](../../results/bw1000/2026-11-28/specialized-sdxl-4792-native-dual-clip-toolchest-independent.json) · [公开组件及校验值](../../results/bw1000/2026-11-28/specialized-sdxl-official-components-transfer-inventory.json)

## 加速执行顺序

| 并行路径 | 先交付的证据 | 继续训练的依据 |
|---|---|---|
| 已有权重先验收 | Gemma 保留已封存的 52/96 对 40/96，并修复续训第 21 步差异。InternVL 核本次冻结的 192 个配对推理槽。MiniCPM、Z-Image、HunyuanVideo、VGGT 各完成本表规定的小规模对照，记录未跑及失败分母。 | 有领域提升且通用保持达标则补连续恢复和独立加载；小样本没有明确结果时，先用开发集选择数据扩充或替补方向，再冻结另一批独立留出。 |
| 已有数据先跑更新 | Qwen 保留真实 OCR 可见答案和逐图内容，补原生首个有效更新。LLaVA 完成末片原权重校验后用既有 64 行 CLEVR 验证原生首步，再加真实图像的对象存在反例。MiniCPM 用新商品家族增加可读规格与遮挡对。 | 每项核实启动命令、权重校验、输入 mask、正学习率和参数实际差，达到条件就进原 25 步和恢复测试。 |
| 共享已授权公开资产 | 商品家族先冻结清单，再从同一原资产产出 SDXL 的单图及文案、HunyuanVideo 的 480p 转台帧、TRELLIS.2 的网格与条件视图、SF3D 的可见 PBR 对。先做一种金属和一种非金属各一份真实输入回执，数据由 CPU 准备。 | 图像/视频以生成质量评估，三维以形状和材质真值评估。共享原始几何和渲染配置，分别验证原生输入。SDXL 完成 VAE 与双文本编码，TRELLIS.2、SF3D 分别验证稀疏缓存和材质缓存后再计训练储备。 [Poly Haven 资产许可](https://polyhaven.com/license) · [ABO 官方版本及署名条款](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) |
| 动作采集并行 | HY-Motion 单开动作数据线，从发布方带标定的 Walking 原记录选新候选，优先核实际标记语义和骨架。再按表演者隔离，复核 128 帧、201 维原生缓存及 40 mm 几何门槛。 | 原生缓存通过后跑一次有效 LoRA 更新，再开始完整窗口；动作标定失败保留原因并换同类合格来源。 [ACCAD 按动作及标定发布的目录](https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data) |

训练卡优先给具备权重、数据和节点放置的真实作业。每启动一项就准备下一项，缺少的数据或模板核验优先走 CPU，业务通过准入后替换所需节点上的保底作业。日报逐底座至少记录独立训练来源数、已证实的有效更新、恢复、盲测分母和新进程加载；单列五项可提交储备及领域交付分数。首批保持每个底座一个候选，同一个底座的额外变体在十二个底座各有交付后再计入累计 50 项。

- **InternVL 盲测准备已完成。** 新集合独立复核 96 张 JPEG、48 条查询和 48 条判断。排查 71 张训练或工程保护图、96 张旧评测图，以及新样本之间的同来源和近重复。公开推理输入已本机打包并逐成员回读，共 98 个文件，归档中没有答案及私有谱系。尚待目标原生预处理和执行两臂推理。 [新盲测复核](../../results/bw1000/2026-11-28/specialized-internvl-cp25-fresh-confirmatory-val96-freeze.json)
- **SDXL 两套文本编码器已在本机闭合。** 原发布版本的两套配置和权重均已按发布方校验，权重共 3,270,967,432 字节。UNet 复用已验证的集群文件，VAE 沿现有中转继续。两套编码器待集群校验和实际条件编码。 [公开组件的本机回执](../../results/bw1000/2026-11-28/specialized-sdxl-official-text-encoders-local-verified.json)
- **OCR 扩样。** TextOCR 论文明确说明标注源自 TextVQA 图像。Qwen 合并两类监督前用原图 ID、来源和近重复共享分组，再拆训练、开发和留出，避免同图以 OCR 和问答两种形式跨集合。保留可见行文本和文字位置训练小字、多行两类，优先复用现有模板。 [TextOCR 原论文](https://openaccess.thecvf.com/content/CVPR2021/html/Singh_TextOCR_Towards_Large-Scale_End-to-End_Reasoning_for_Arbitrary-Shaped_Scene_Text_CVPR_2021_paper.html)
- **室内位姿扩样。** 若 VGGT 原 household 对照需要更多训练轨迹，开发阶段可以验证 TartanAir v2 的图像和相机姿态。先转换左右手坐标及相机外参约定，看真实同输入能否消费；按合成场景和序列分组，保留 TUM 的真实序列作为独立留出。 [TartanAir 发布方](https://tartanair.org/) · [TUM 原始基准](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf)
- **InternVL 集群入口。** 两份源码已发布于 `035bbe258`。原公开输入包在 SSH 写入阶段超时，发送方未收到完成写入或接收确认。远端只读核查未见原目标目录、原输入清单和收件回执。原请求保留待对账，原生 CPU 预处理和两臂推理还未启动。 [原上传和对账](../../results/bw1000/2026-11-28/specialized-internvl-cp25-confirmatory-original-upload-timeout-and-reconciliation.json)
- **同一素材覆盖图像和材质输入。** 漆木凳已就地生成 256×256 棚拍 RGB、RGBA 和可见像素材质，复核 12,457 个前景像素的掩膜和色值一致，粗糙度有变化，金属度均为零。作为区别于工具柜的非金属家族候选，下一步分别进入 SDXL 原生编码和 SF3D 可见点条件验证。 [漆木凳派生及核对](../../results/bw1000/2026-11-28/specialized-painted-stool-cc0-sdxl-sf3d-fixed-view-independent.json)
- **SDXL 原生 VAE 已验证。** 集群端逐片校验 80 个分片，并核对完整官方权重及配置 SHA256。零卡步骤 `1103606.4815` 用目标 Diffusers 0.39.0 载入冻结的 FP32 VAE，工具柜和漆木凳分别得到可重复采样的 `[1,4,32,32]` latent，两幅图的 latent 校验值不同。两套官方文本编码器均已开始按独立分片上传。完整条件编码、UNet 更新和恢复仍待验。 [VAE 权重及实际编码](../../results/bw1000/2026-11-28/specialized-sdxl-official-vae-two-cc0-native-latents-4815.json)
- **LLaVA 原生加载的当前边界。** 两次早期检查分别定位到 ModelScope 只读目录建锁、Transformers 的缓存类型导入。改为进程私有缓存并接回原已通过的兼容层与视觉依赖后，`1103606.4818` 的真实加载进度到了 698/698，随后在原 202 秒进程时限终止。仍需证实加载器返回及 504 个 LoRA 项实际初始化，训练准入待验。 [实际超时与加载进度](../../results/bw1000/2026-11-28/specialized-llava-native-4818-698-weight-progress-then-timeout.json)
- **LLaVA 已确认加载器和 LoRA 构造返回。** `1103606.4819` 进一步定位到权重读取后的 Transformers 初始化，原生 Swift 加载边界耗时 289.29 秒，LoRA 构造耗时 0.73 秒。进程在原 504 项 LoRA 契约的合并断言上失败。下一次检查需分别报告目标模块和参数名称、形状、dtype 及数量差异，并核对实际权重分配的缺失或意外键。当前没有有效更新。 [原终态、加载与 LoRA 证据](../../results/bw1000/2026-11-28/specialized-llava-native-4819-loader-returned-lora-contract-failed.json)

## 第一批交付的接续顺序

1. **Gemma、InternVL、MiniCPM 优先完成已有检查点的验收。** Gemma 的两个 CP25 适配器文件已在 SCNet 内复制并重新打开校验，合计 119,281,889 字节。独立领域盲测封存为 52/96 对 40/96，仍需解决恢复后第 21 步差异。TextVQA 通用保持已有完整的 5,000 道无答案问题、3,166 张图片的发布方索引和 3 张已在目标核验的图片，后续缺的是完整图片、训练重叠排查及同题 A100 对照。InternVL 只继续对账现有 96 题交接，已训练的 CP25 与复载结果可直接复用。MiniCPM 仍保留 28 条封存预测，按七个独立商品组报告数值单位联合正确和不可读误答。 [Gemma 候选权重回执](../../results/bw1000/2026-11-28/specialized-gemma-cp25-candidate-copy-and-readback.json) · [TextVQA 原问题与媒体回执](../../results/bw1000/2026-11-27/training-gemma-cp25-textvqa-val5000-question-plan-3-public-images-and-native-source.json)
2. **Qwen、LLaVA、SDXL、SF3D 争取尽快取得有效新更新。** Qwen 对账原八卡请求；LLaVA 使用已发布的键与 LoRA 分项检查，核准后在冻结的 CLEVR 行做原生 eager 首步；SDXL 用同一工具柜和漆凳分别绑定 VAE、双文本条件、噪声及 timestep；SF3D 先使工具柜前景像素上的可见 XYZ、冻结特征和材质监督严格对应。首步各自记录正学习率、梯度和参数变化，通过后接连续更新与恢复。LLaVA 真实图像的独立筛选可增加 POPE 的存在性问答，与原自由描述 CHAIR 一并观察，保留有物体漏报和无物体误报。POPE 及其原图不纳入训练。Qwen 的陌生来源可考虑 ST-VQA，实际选入前逐图检查许可及与已有 TextVQA、TextOCR 的重叠。 [LLaVA 诊断源码](../../tools/llava_recovered_native_cpu.py) · [POPE 原论文](https://aclanthology.org/2023.emnlp-main.20/) · [ST-VQA 发布方](https://rrc.cvc.uab.cat/?ch=11)
3. **HunyuanVideo、Z-Image、VGGT 先兑现原检查点。** HunyuanVideo 先闭合合法单卡的原检查点重载和清场，再运行三件商品的两臂预筛；Z-Image 在生成前给两名独立评审准备盲序，锁定各 16 个底座与适配器输出；VGGT 在原载体可用时完成 20 个冻结槽的两臂评分，19 个几何有效槽之外的失败继续计入分母。通过小样后扩足独立家族或录制序列。
4. **HY-Motion、TRELLIS.2 优先形成首条原生样本。** HY-Motion 使用同源且已标定的真实动作，保留按表演者隔离和严格小于 40 mm 的几何门槛。AMASS 可作为含 ACCAD 和 CMU 的补充来源，需核对其分发许可、身体模型许可、转换精度及原表演者重叠后采用。TRELLIS.2 先对工具柜生成正式 O-Voxel shape latent 和同物体渲染条件，让实际 Shape Flow 入口复读坐标、特征及 1024/2048 token 预算。已有四件 Khronos 留出继续隔离。 [AMASS 发布方](https://amass.is.tue.mpg.de/) · [TRELLIS.2 官方数据工具](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md)

本轮新增两组可复读的公开 CC0 素材。录音机 8 个文件共 944,682 字节、电水壶 5 个文件共 2,387,109 字节，glTF 完整文件闭包、发布方校验值及本地独立回读均通过。它们分别是电子设备和厨房家电候选家族，在域内划分冻结后，可派生 SDXL 单图、HunyuanVideo 桌面转台、TRELLIS.2 网格与条件视图。两件各有 3 套、2 套材质，进入 SF3D 前先核其多材质可见点监督。新增 2 个源物体，当前新增 0 条原生训练行，0 个正式领域交付。 [录音机](https://polyhaven.com/a/cassette_player) · [电水壶](https://polyhaven.com/a/vintage_electric_kettle) · [发布方 CC0 条款](https://polyhaven.com/license)

- **LLaVA 已过真实权重和 LoRA 核验。** 步骤 `1103606.4860` 以 307 秒正常退出。官方权重在真实加载分配中没有缺失键、意外键或不匹配键。旧候选与参考正则的选项顺序不同，实际都匹配同样的 252 个模块，504 个 LoRA 张量的名称、形状和 dtype 全部一致。复用 `1103606.4719` 已验的 64 条原生 CLEVR 监督行，下一项合并到统一训练入口并派发八卡首轮有效更新。两份现有证据均为零卡准入。 [真实加载及 LoRA 证据](../../results/bw1000/2026-11-28/specialized-llava-4860-full-native-checkpoint-and-lora-contract.json) · [64 条原生监督](../../results/bw1000/2026-11-28/specialized-llava-4719-clevr64-native-supervision-independently-verified.json)
- **HY-Motion 保留 B03 的一次真实失败。** C3D 的 41 个标准标签与独立参考一一相符，额外的 `Male1:LFRM-1` 已单独保留。实际拟合 RMS 为 9.2205 mm，独立留出仅 21/92 帧有效，低于原要求 79 帧。沿用严格小于 40 mm 的门槛，改选有独立标定与同步记录的动作，B03 不进入训练。 [原标签和一次标定终态](../../results/bw1000/2026-11-28/specialized-hymotion-accad-b03-41-label-projection-and-independent-calibration.json)
- **SDXL 第一套文本编码器已在目标集群组装。** 118 个分片的大小和 SHA256 均逐片核验，492,265,168 字节权重的完整 SHA256 与发布方一致。第二套继续按分片落盘。两路真实条件齐备后再绑定工具柜与漆凳的既有 VAE latent。 [组装回执](../../results/bw1000/2026-11-28/specialized-sdxl-first-text-encoder-official-118-parts-target-assembly.json)
- **LLaVA 已在目标端绑定首训候选。** 首个张量校验 `1103606.4877` 在完成 64 行后因设备句柄检查失败，保留失败账。去掉容器加速卡注入后，`1103606.4878` 正常完成，64 个图文样本全部复核通过，原生 token、标签、掩膜和图像张量指纹仅留目标端。统一 Swift 首训候选已发布并在目标端复读核对 18 个成员。按原 25 步调度走前两个 AdamW 边界，第二步必须为正学习率且参数实际改变。八卡接管、实际更新、CP2 和独立清场仍待执行。 [64 行实际终态](../../results/bw1000/2026-11-28/specialized-llava-4878-all-64-native-collator-tensor-witness.json) · [目标首训暂存](../../results/bw1000/2026-11-28/specialized-llava-clevr-first2-18-target-files-and-admission-stage.json)
- **LLaVA 两次八卡步骤均已终结并清场。** `1103606.4884` 的 Torchrun 参数分隔问题已在目标容器用零卡步骤 `1103606.4886` 验证修复。修正版 `1103606.4890` 启动八 rank，在 OneVision 模型模块导入时触发目标 Transformers 5.5.0 的缺失符号。原插件保护条件只接受 5.12.1。两次都保留 64 个计划前向和 16 个 rank 更新边界的分母，实际更新为 0，独立清场均通过。下一版先验证显式 eager 配方下只在目标原模块导入期间适用兼容保护，保持 Flash 导出和原模型配方。 [Torchrun 终态及修复](../../results/bw1000/2026-11-28/specialized-llava-4884-torchrun-failure-clean-and-option-fix.json) · [八 rank 导入终态](../../results/bw1000/2026-11-28/specialized-llava-4890-eight-rank-vision-import-failure-and-clean-post.json)
- **LLaVA 的 eager 导入修复已在目标容器复核。** 零卡步骤 `1103606.4896` 在原模型文件上复现旧失败，再成功导入修正后的 OneVision 类，并恢复原运行时状态。修正版首训包 19 份文件已在目标端逐项读回，沿用同一份 64 行训练指纹。下一次八卡派发前，核原 Swift CLI、参数、回调和节点归属。 [零卡配对导入](../../results/bw1000/2026-11-28/specialized-llava-4896-paired-real-eager-import-target.json) · [修正版暂存回执](../../results/bw1000/2026-11-28/specialized-llava-eager-fixed-19-target-files-qualified-staging.json)
- **LLaVA 的原 Swift 首训入口通过零卡检查。** `1103606.4897` 正常退出，真实 `SwiftSft` 入口、`SftArguments`、89 个参数 token、OneVision eager 类和原训练回调全部核验。在训练参数初始化前截停，实际前向和优化器更新均为 0。八卡控制入口已绑定这 19 个成员；派发前仍须闭合上一条保底请求的归属并执行新现场检查。 [Swift 原生入口与独立 Slurm 记录](../../results/bw1000/2026-11-28/specialized-llava-4897-native-swift-cli-and-callback-zero-gres.json)
- **LLaVA 第三次八卡首训推进到原生训练循环。** `1103606.4901` 的八个 rank 都收到整理好的图文输入，在首轮模型调用的观察回调处失败。实际 Swift 多模态 hook 先把 `input_ids` 转成 `inputs_embeds`，原审计 hook 后运行时已经取不到 `input_ids`。将审计 hook 提前，保留原 Swift 的嵌入、前向和损失路径。作业记录为 0/64 完成前向和 0/16 rank 更新，未保存 CP2，独立 `1103606.4902` 清场完成。 [八卡失败与清场](../../results/bw1000/2026-11-28/specialized-llava-4901-swift-input-ids-loss-failure-and-clean-post.json)
- **LLaVA 的 hook 顺序修复通过目标端复现。** 首个复现实验脚本因首行多余运算符退出。修正版在 `1103606.4908` 的真实 Swift/PyTorch 容器中复现原 `KeyError`，提前审计后检查到同一输入，且 Swift 模块输出与原流程一致。此处使用受控张量验证调用顺序；下轮仍须实测模型前向、有限 loss 和第二次 AdamW 更新。`1103606.4906` 的八张卡已恢复保底，两轮采样的计算计数均增长。 [目标端配对和节点回执](../../results/bw1000/2026-11-28/specialized-llava-4908-hook-order-pair-and-4906-standby-return.json)
- **LLaVA 新作业 `1103606.4912` 在配置类导入时退出。** 一个子进程没有从动态模块读到原 OneVision 配置类，八个 rank 均未到训练开始标记。没有真实更新或 CP2，`1103606.4913` 独立清场通过。目标 Transformers 向共同的动态模块缓存复制本地源码，锁的作用域限于进程内。新版在 Transformers 导入前指定各 rank 独立缓存，`1103606.4918` 的八路并发验证均读到与原文件逐字节一致的实际配置和模型类。新的八卡首训仍需验证有限 loss 与参数变化。 [修正包实际派发](../../results/bw1000/2026-11-28/specialized-llava-4912-native-training-prehook-eight-card-dispatch.json) · [实际终态和清场](../../results/bw1000/2026-11-28/specialized-llava-4912-dynamic-config-import-failure-and-4913-clean-post.json) · [八路并发导入实证](../../results/bw1000/2026-11-28/specialized-llava-4918-eight-independent-native-dynamic-imports.json)
- **Gemma TextVQA 媒体查重已核对 27 张图。** 先对已有三张官方 JPEG 核验，再通过发布方 ZIP 的精确范围取得 24 张。合计新增验证口径覆盖 40 道题，各媒体的编码字节 SHA 均与原 64 张训练图不重合。还需核验 3,139 张图和 4,960 道题所对应的媒体，正式保持性仍未评分。 [三张确证](../../results/bw1000/2026-11-28/training-gemma-cp25-textvqa-val5000-three-media-original64-overlap.json) · [24 张官方 ZIP 新媒体](../../results/bw1000/2026-11-28/training-gemma-cp25-textvqa-val5000-publisher-range-24-additional-images.json)
- **LLaVA 的独立缓存包已核对 19 件。** `1103606.4919` 的控制步骤发出有序退出信号，原保底 `1103606.4917` 正常退出。随后一次 Slurm 查询未通过校验，控制步骤结束，未提交 GPU 训练。`1103606.4920` 独立复核同一 boot 和八 PCI、设备句柄与端口后，释放本地占用。更新后的交接入口在原定释放期限内等待下次有效查询，继续要求精确的 Slurm 终态和独立清场。 [目标暂存](../../results/bw1000/2026-11-28/specialized-llava-rank-private-cache-19-member-target-stage.json) · [交接终态和清场](../../results/bw1000/2026-11-28/specialized-llava-4919-prelaunch-query-failure-and-4920-clean.json)
- **Gemma 已累计核验 379 张图。** 新增两批 96 和 256 张来自发布方 ZIP 的原始 JPEG。已核媒体覆盖 712 道正式留出题，逐图 CRC、SHA256 与 JPEG 解码通过，且未与原 64 张训练图发生编码字节重合。剩余 2,787 张图和 4,288 道题仍待核验。正式 TextVQA 分母保持 5,000，无新评分。 [96 张批次](../../results/bw1000/2026-11-28/training-gemma-cp25-textvqa-val5000-publisher-range-96-additional-images.json) · [256 张批次](../../results/bw1000/2026-11-28/training-gemma-cp25-textvqa-val5000-publisher-range-256-additional-images.json)
- **Gemma 已累计核验 891 张图。** 第三批从发布方 ZIP 新取得 512 张图，累计覆盖 1,641 道题，逐图通过 CRC、SHA256 和 JPEG 解码且无原训练字节重合。2,275 张图还需核验，正式保持性仍未评分。 [512 张官方 JPEG](../../results/bw1000/2026-11-28/training-gemma-cp25-textvqa-val5000-publisher-range-512-additional-images.json)
- **SDXL 续传第二套官方编码器。** 在目标端确认 255/663 个分片后，已从第 256 片接续。最近逐片回执至少确认 454/663 片，第一套已通过整件校验。完整回执到齐后，组装命令依次核对逐片哈希、整件哈希及整件读回，再发布权重供双文本条件生成。已验证组装入口能拒绝缺片、损坏片和重复发布。

## 加速交付的执行顺序

先让十二个底座各出现一个独立可计分的候选。已有权重的评测、短训首步和数据准备并行，准备工作使用 CPU。训练节点仅在真实配方及放置就绪后按需接替保底。每个底座先做出一份可复验候选，再给同底座开第二个领域方向。

| 执行顺序 | 底座 | 本轮要拿到的证据与下一组数据 |
| --- | --- | --- |
| 已有权重，优先验收 | Gemma | CP25 继续作为交付候选，保存已揭盲的 96 题结果。新采 USGS 事件用于独立验证，按事件 ID 分组冻结。先定位恢复后第 21 步参数差，再推进 TextVQA 和 CIDEr 保持性。 [USGS 官方目录](https://earthquake.usgs.gov/fdsnws/event/1/) |
| 已有权重，优先验收 | InternVL | 原生 CP25 复载资格可直接复用。原 96 题交接仅对账原请求，接收已证实后跑 192 次生成。现有 96 张图按 48 条关系查询、48 条真假判断评分，保留原冻结阈值。 [CP25 新进程复载](../../results/bw1000/2026-11-28/specialized-internvl-4809-cp25-fresh-native-reload-504-lora-exact.json) · [独立盲测冻结](../../results/bw1000/2026-11-28/specialized-internvl-cp25-fresh-confirmatory-val96-freeze.json) |
| 已有权重，优先验收 | MiniCPM | 沿用封存的 28 条预测和 7 个商品组。独立评分后，再按品牌及 SKU 家族扩充可读、不可读各 48 道盲题。训练补充样本先从 Open Food Facts 已核出处实际照片中选，读不清的标注由人工核实。 [原封存和清场](../../results/bw1000/2026-11-27/specialized-minicpm-OFF-4495-paired14-sealed-and-4528-clean-post.json) · [发布方数据条款](https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/tutorials/license-be-on-the-legal-side/) |
| 首轮真实更新 | LLaVA | 八卡窗口只在完整入口和资源核实后接续。固定 64 个样本做两步，第二步必须正学习率且实际参数变化。CP2 新进程复载后，开发集查漏报与幻觉，再用从未训练的真实图片评价 CHAIR 和 CIDEr。POPE 作为独立的存在性测量。 [CLEVR 发布方](https://cs.stanford.edu/people/jcjohns/clevr/) · [CHAIR](https://aclanthology.org/D18-1437/) · [POPE](https://aclanthology.org/2023.emnlp-main.20/) |
| 首轮真实更新 | Qwen | 原八卡请求明确后，复用 41 条原生输入启动首个正学习率更新。下一批 TextVQA/TextOCR 按共同原图 ID 排除交叉，优先小字、多行、反光均有的可读图。独立留出首轮预算 96 张图，使用 TextVQA 原评分并分层报告。 [TextVQA](https://textvqa.org/dataset/) · [TextOCR](https://textvqa.org/textocr/dataset/) |
| 已有权重，验证加载与效果 | HunyuanVideo | 单卡复核 216 个 LoRA 张量并独立清场，再做 3 个未见商品的两臂预筛。后续用独立商品家族扩展。只复用相机轨迹，用各真实物体自己的参考帧检查主体、闪烁和 Video CLIP。 [25 步训练原件](../../results/bw1000/2026-11-27/specialized-hunyuanvideo-abo-v3-4560-25-native-updates-and-4561-independent-post.json) · [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) |
| 已有权重，验证加载与效果 | Z-Image | Style18 CP25 按冻结盲序生成 16 对。独立评审按风格与结构同时合格率判断，并保留通用提示和失败。若这一路线无法形成可复核盲评，准备 Met 已标记 Open Access 的陶瓷器形与纹样，用年代和器形族做留出，采用原 Differential LoRA。 [The Met Open Access](https://www.metmuseum.org/hubs/open-access) |
| 已有权重，验证加载与效果 | VGGT | 使用原冻结室内 20 槽，对原底座和 CP25 作同题配对，未过几何的一槽计失败。取得首轮结果后按完整录制划分新 TUM 序列，禁止相邻帧跨训练与盲测。 [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) |
| 数据与首轮真实更新 | SDXL | 两套编码器整件核验后，立刻在工具柜、漆木凳生成固定双文本条件并绑定已验 latent、噪声和 timestep，使用原 UNet LoRA 获取真实更新。素材扩充优先用已有网格闭合的异家族商品，背景和视角不跨划分复用。 [VAE 原生成](../../results/bw1000/2026-11-28/specialized-sdxl-official-vae-two-cc0-native-latents-4815.json) · [Poly Haven CC0](https://polyhaven.com/license) |
| 数据与首轮真实更新 | SF3D | 工具柜 20,600 个可见点已核坐标、三角面、UV 和三类 PBR 值。原接口读取图像级标签，工具柜和木凳均存在空间变化。优先用 CC0 网格渲染已知恒定 PBR 的原生相机视图，适配统一数据读取并保留原 36 张量训练。空间变化的材质单独走可见点监督，先验证同视角 triplane 采样。盲测按未见资产族留出，单列三项 MAE。 [点级与训练接口实证](../../results/bw1000/2026-11-28/specialized-sf3d-metal-tool-chest-point-contract-and-head-compatibility.json) |
| 数据与首轮真实更新 | TRELLIS.2 | 优先交出工具柜的一条同物体条件与 O-Voxel shape latent，实际过原生 Shape Flow loader。录音机、电水壶可补不同家族，已冻结的 Khronos 四件留出保持隔离。 [官方数据制备](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md) |
| 数据与首轮真实更新 | HY-Motion | ACCAD 只选同时具备原轨迹、标定和时间覆盖的动作。先按同一几何门槛取得真实 128 帧、201 维缓存。并行核 CMU 的普通行走、转向和停止，Subject 91 的行走转向与 Subject 128 的停止作为来源候选。比较原标定和时间支持后再选择，不能按最终动作评分挑数据。 [ACCAD 发布方](https://accad.osu.edu/research/motion-lab/mocap-system-and-data) · [CMU 动作索引](https://mocap.cs.cmu.edu/subjects.php) |

小样本用于决定是否继续当前方法。正式盲测按图像、事件、商品族、动作表演者或录制序列冻结，使用同输入和同随机种子，对原底座和候选保留失败、缺失及生成上限。每个模型提交可重载权重、数据与底座版本、训练恢复记录和逐项分数后，才增加领域模型交付数。领域提升与原 17 项 A100 性能和保持性验收分别计算。

## 下一批实际执行

- **Gemma** 优先修复已观察到的 CP20 恢复第 21 步差异，并用同题独立盲测复用已封存的 52/96 对 40/96。恢复和通用保持补齐后核验候选包。
- **LLaVA** 已有八卡真实更新及 CP25。目标 CPU 原生前向已确认视觉首轮带图、第二轮复用历史且不再传图，缓存长度为 0→227→228。正在将这条修复接回冻结的 Val16 两臂 GPU 盲测，准备后才启动正文生成。 [两次真实原生前向](../../results/bw1000/2026-11-28/specialized-llava-5226-native-cache-fix-227-prefill-one-decode-zero-gres.json)
- **SDXL** 四条真实图像行已过目标原生读取。下一项严格执行 25 步连续 + 从本次 CP20 在新进程恢复 5 步，检查真实样本游标和最终状态。现有六对图片先完成独立盲评，保持三件盲测资产隔离。 [四行训练作业范围](../tasks/training-sdxl-cc0-four-full25-cp20-r33-owner-v1.json)

- **SDXL 已取得真实训练步骤编号。** `1194900.2769` 在目标端实际运行，一张 DCU、8 CPU。独立零卡准入 `1194900.2767` 与 owner 前检 `1194900.2768` 均正常终结，等待连续训练与恢复的更新账。 [首次运行证据](../../results/bw1000/2026-11-28/specialized-sdxl-2769-four-row-native-full25-cp20-first-numbered-running.json)
- **LLaVA Val16 修复已并入新执行包。** 冻结 16 个图文输入和 32 个输出槽，原底座与 CP25 各占 16 个槽。实际 GPU 执行必须同时验证基础权重、适配器、修复命中、完整分母和独立清场。 [配对盲测任务](../tasks/training-llava-clevr-val16-native-cache-v4.json)

- **SDXL 四行训练与恢复已实际通过。** `1194900.2769` 正常结束。独立验收确认 25 步连续训练与 5 步新进程恢复、四份检查点、CP20 第 4 个 epoch 已消费 4 条的游标，以及恢复末态与连续训练末态完全一致。`1194900.2770` 独立清场通过。新 CP25 的领域盲测待跑。 [独立终态](../../results/bw1000/2026-11-28/specialized-sdxl-2769-four-native-rows-25-plus-5-exact-CP20-and-clean.json)
- **LLaVA 盲测已在目标端就绪。** 16 份原图和原题已复核，32 个输出槽已冻结。上次读到 n14 仍有八卡保底步骤 `1103606.5220`，须完成有证据的接管后开跑。 [目标暂存与当前节点边界](../../results/bw1000/2026-11-28/specialized-llava-val16-v4-answerless-target-stage-and-n14-existing-standby.json)

- **SDXL 新 CP25 已重新读回。** 连续和恢复的适配器均为 11,778,368 字节且内容相同。新六槽生成器固定原有提示、噪声和时间步，复用 12 张对照图，优先比较四行训练对两行训练。新六图待 GPU 执行。 [检查点证据及对照范围](../../results/bw1000/2026-11-28/specialized-sdxl-2769-continuous-cp25-adapter-pins-and-new-six-slot-plan.json)
- **HY-Motion 已验证一条真实左转试次的根轨迹。** ACCAD Male2 B9 用前 64 帧拟合原 120Hz C3D 和 30Hz BVH，后 64 帧完整、独立 RMS 25.60 mm，低于 40 mm。原 41 点读取器拒绝该表演者的 82 点校准文件。Male2 继续留在开发组，完成标记绑定和 201 维检查后，再将方法应用到独立训练表演者。 [实际原件和结果](../../results/bw1000/2026-11-28/specialized-hymotion-accad-male2-b9-first-independent-root-geometry-and-calibration-roster.json) · [ACCAD 公开数据](https://accad.osu.edu/research/motion-lab/mocap-system-and-data)

- **LLaVA 的合成场景盲测已实际结束。** `1103606.5232` 在单卡上完成全部 32 个生成，底座和 CP25 各 16 个，全部达到原 128 token 上限，均无 EOS，按冻结口径每臂 **0/16** 可进入内容评分。修复通过了生成与独立清场，领域改善仍无证据。 [独立终态及截断汇总](../../results/bw1000/2026-11-28/specialized-llava-val16-v4-5232-two-arms-all-cap-and-clean.json)
- **Gemma 下一项针对停在首前向的恢复差异。** 新源码只观测全局 20 步后的第一个微批，在 rank 0/1 记录前后 RNG、训练上下文和既有损失摘要。保持原 8 rank、4 次累积和 25 步调度，到实际更新 21 停止。本地已验，目标仍需新启动范围。 [诊断入口](../../code/bw1000/training-gemma-usgs72-resume21-light-witness-v1/HANDOFF.md)

- **SDXL 配对盲测图已备齐。** `1194900.2821` 用四行训练的 CP25 生成 6 张新图。独立终态和 `1194900.2822` 清场通过。复用原底座与两行 CP25 的 12 张对照图，核实六组原噪声和时间步一致。18 张图待评分，领域验收保持待定。 [独立生成与配对证据](../../results/bw1000/2026-11-28/specialized-sdxl-2821-four-row-cp25-six-images-and-three-arm-index.json)
