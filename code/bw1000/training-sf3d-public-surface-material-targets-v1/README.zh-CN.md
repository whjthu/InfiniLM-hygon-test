# SF3D：真实表面积测度的全局 PBR 目标输入

本轮补充的是已下载公开对象的**材质标签定义和原生字段接线**，不是再做旧 atlas 目标核验或 T10 几何评分。计划 WaterBottle/Corset 两训练组、BoomBox 一开发组；当前实际完成 WaterBottle+BoomBox 两对象，Corset 源访问被拒，完整 cohort 未通过。

## 新的实质变化

既有公开 PBR recipe 的 `native_batch_*` 是完整 atlas texel 均值；原256 UV点是另留的探针，不能把二者混称。未使用或密集铺开的 UV 区域可能改变 atlas 均值。

新独立 recipe `sf3d-public-surface-area-nearest-PBR4096-v1` 使用实际 glTF POSITION/index/TEXCOORD_0：按三角面积分布选择面，再用 sqrt 重心法取4096个表面点并插值 UV，从原 baseColor/ORM 取材质值。取样 seed 由对象名与固定 recipe 字符串确定，先于本次纹理读取，不依据模型/答案选样。

保留原 `pbr.py` 中已经冻结的 `decode_pixels`，不重写色彩公式：albedo 解 sRGB 再乘 factor；roughness 取 ORM 绿通道，metallic 取蓝通道及相应 factor。原纹理过滤未声明，所以**本 recipe 明确选择 base-level nearest texel/REPEAT**，不是宣称匹配某个 GPU shader 的默认 min/mag/mipmap 过滤。

源仅接受固定静态 OPAQUE、TEXCOORD_0、无 texture transform、均匀正 scale 的这些对象。面积比例在已核的统一 scale/TRS 下不变。退化面、索引或完整 UV 不合即失败，不删面、修洞或造材质。两个真实对象各4096槽，保原4510/6036面来源；不是4096个新独立对象。

实际差异如下，只表示**两种测度定义不同**，不是误差下降/模型改善：

| 对象 | albedo最大绝对变化 | roughness绝对变化 | metallic绝对变化 |
|---|---:|---:|---:|
| WaterBottle | 0.06372179 | 0.03034580 | 0.18533871 |
| BoomBox | 0.00618279 | 0.02508456 | 0.03000795 |

这些是固定4096样本的 Monte Carlo 估计，没有完整表面积积分误差界，不冒精确物理真值。

## 与真正共同入口的连接

冻结 `generation/src/gencuda/backends/sf3d_training.py` 58458B / `dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802` 的 cold-cache 路径在1110行生成每图全局 target，1127–1129行存 albedo(1,3)、roughness(1)、metallic(1)。

`load_native_targets(path, exact_pin, RECIPE)` 已实际读回新 NPZ，检查完整4096采样身份和三个 float32/[0,1] native 字段；`to_original_target_object(fields)` 已实测返回该入口接受的 albedo(3) ndarray、两个 Python float。未来独立 domain 分支只能在**targets 供料位置**接入，保原36张量/2644871参数、特征缓存和 optimizer 数学不变。原8RGBA heuristic 和旧 atlas recipe 均不被替换。

本包**尚未部署到 native caller**，只给可调用的数据适配函数。没有产生 `triplane_features` 或 `image_features`，也没有把4096标签采样点写成原 native 随机3D query points。原模型、原8RGBA、权重/检查点审批范围均未接触。

`training_identity(rows)` 仅在两个真实训练对象 WaterBottle/Corset 都完成时返回新 recipe+mesh+texture+target 的有序身份。当前只有 WaterBottle 可用，已经实际拒绝生成完整训练身份；不能拿 BoomBox 开发组补训练。后续还须将该标签身份与原 native config/input-image 身份共同绑定，以免复用旧 atlas labels 的 cache/resume。现在没有任何真实模型/cache/resume资格。

## 本轮实际范围和阻塞

本机0.406秒完成2对象/8192真实表面槽、两份真实目标 NPZ，所有字段有限、形状/范围正确。额外独立几何 oracle 只有两块面积1:4的三角面，8192固定样本的大面比例0.79736328125，通过预注册0.02带；它不是数据/模型效果。

读取 Corset 的精确 `local-only/training-sf3d-public-corset-train-atlas-v1-original/Corset.gltf` 2487B / `1aaa7fa54731ca833c3a0f919ba82c1af2a69cc7f8de36ba76bfb5038de8bd38` 实际 PermissionError。本轮按“不请求权限”没有提权、重试、复制或改路径绕读。一次拒绝发生在实现前精确 metadata 检查，正式 `check_existing_once.py` 明确跳过该已拒文件，保计划3/完成2/阻塞1；其退出0只表示预注册的部分核验完成，结果 `passed=false` 表示完整 cohort 未过。

根最短合规路径：在自己已有可读权限范围核原 Corset glTF 和已有 base/ORM，不重新下载；如本机同样不可读则继续阻塞。相应原公共 metadata 仅列身份，不读取被拒正文：

- `Corset_baseColor.png` 4501246B / `76a4c45762a26e3bc8a62ea72cf791bd376d310b7bb0856daecac2278c00e626`
- `Corset_occlusionRoughnessMetallic.png` 3443675B / `c3b41efc3a39fd2b3b9d6f0380057a826192632c4a099f4e3fa0452b8dfce0dc`
- 已下载 `Corset.bin` 662184B / `de7730fb4201466603410dc1a7128bc1339e945a945dbb852f6dc39f3c53ec22`

当前 `SPECS` 只含本轮可读两个对象。根以后可在新独立 scope 对第三对象调用通用 `surface_samples`，复用此轮两份已核目标，不重跑它们。还需把新 target-only 供料真正接 native caller，使用真实图像/特征、模型、完整状态和设备准入；输入仍是 atlas，物体视图/特征语义适用性未证明。原有训练/开发/盲测分组不改，未打开盲答案。

## 来源与交付

这些原对象素材依各自 [WaterBottle](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/WaterBottle/LICENSE.md)、[BoomBox](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/BoomBox/LICENSE.md)、[Corset](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Corset/LICENSE.md) 原声明 CC0，文档 CC BY4，保原作者/修改说明。材质/UV定义参照 [glTF 2.0 官方规范](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)。上游完整 commit 仍未确认，沿实际每文件 SHA 固定。

raw纹理、材质值、采样坐标和 NPZ 只留忽略 local-only。Git白名单只有自编源码、说明和小证据，由根串行处理。本轮0网络/权限请求/集群/SSH/GPU/optimizer，新增 READY=0。原两对象不足、三对象atlas及原T11 FAIL/UNKNOWN全部保留，不用新标签冒原正式17验收。
