# T11 独立公开PBR材质输入：1/2对象完成，整包失败

本轮先查T07真实原入口：它在已执行的CSV/LoadImage之后必须调用torchvision.resize(BILINEAR)+center_crop。本机实查Torch/TorchVision均无，故未替代、未重复既有constructor，转做本独立T11公开输入。原T11检查点传输审批、原8RGBA及缓存均未读取或修改。

来源是Khronos官方glTF-Sample-Assets的WaterBottle/BoomBox，作者Microsoft2017。实际metadata声明asset内容CC0，LICENSE另将许可/元文档标为CC-BY4；保留原件、来源、署名和修改说明，不把文档也说成CC0，logo/trademark权利除外。[WaterBottle](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/WaterBottle)、[BoomBox](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoomBox)、[许可](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/WaterBottle/LICENSE.md)。实际抓取官方main，各body/ETag冻结；没有取得upstream commit，不能称完整revision校验。以后须复用这些固化副本，不静默重新下载main。

唯一实际请求3425bcd8f8884785b23ef0192e187f39：180秒单epoch、Job实际2CPU/2GiBcommit，10GET/9,037,799正文bytes/32.172秒，C1非timeout。WaterBottle完整2张2048²PNG，RGBA输入及真实PBR参数解析完成；BoomBox只有baseColor成功，ORM声明大小超过4MiB，读正文前拒绝0bytes。保2对象、4PNG、512UV总分母，实际1/2对象、3/4PNG、256/512UV，不涨cap、不重GET、不换对象。开发组失败，盲组0，不是训练READY。

预注册整对象分组：WaterBottle train、BoomBox dev，各对象的全部纹理、patch和将来视图共组。当前仅1训练对象，dev缺材质图，blind无对象；相同作者/制作风格也不构成OOD证明。已完成raw内容去重只覆盖实际收到的3PNG；完整跨组像素去重尚未完成，不凭不同行名宣无泄漏。公开只身份、许可、计数和统计，不提交原图、raw JSON/标签。

`pbr.py` 按[官方glTF规范](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials)解释sRGB baseColor到linear，G为linear roughness、B为linear metallic，乘原factor。标签是作者定义的PBR参数，不是实验室测量；opaque UV atlas并不等于物体表面积或相机可见面。生成16×16固定UV中心点（256slots、不按颜色/答案选），另用全4,194,304 UV texels算global均值，提供明确目标字段albedo(1,3)、roughness(1)、metallic(1) float32。

实际执行冻结共同backend原`foreground_material_targets`的纯NumPy原函数体，未import整个框架、Torch、模型或stub。其原目标是sRGB颜色/饱和度启发式；相同WaterBottle RGBA与PBR UV均值的绝对差为albedo0.198880、roughness0.269379、metallic0.320990。这是监督语义差异，不能当模型MAE或说明哪一个原参考已被改善。新物理标签须独立recipe，不静默覆盖原8例及原A100参考。

`read_targets.load_targets(path,expected,recipe)`是实际可调用的target-only桥：只有显式`sf3d-public-pbr-UV-atlas-v1`、精确NPZ SHA/长度才返回原字段名/形状的NumPy目标。真实落盘包256UV/finite/range/shape已通过，错recipe及错SHA各1次拒绝。另3个新增独立小合成合同验证了官方sRGB例值、ORM红通道无关及失败不丢256分母；不冒真实模型效果。

还缺：实际SF3D预处理/冻结triplane_features及image_features、atlas与真实视图监督定义、dev原ORM、未触碰的blind对象、新配方A100同输入数值/真实更新恢复。没有下载几何BIN/GLB或模型权重；不伪造featurecache，不启动训练或提交Slurm。原T11的36张量/2,644,871参数配方没有改变。

`once.py`已唯一执行，不可重放。源码源码身份在SOURCE_FILES/SOURCE_PROOF；原GET失败与partial全保留。训练/GPU/Model/formal17 GO=false，READY+0。
