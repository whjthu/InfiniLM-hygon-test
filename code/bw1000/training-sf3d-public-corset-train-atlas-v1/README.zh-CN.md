# SF3D 公开材质的第二个训练对象

沿用冻结的 PBR UV 标签公式，将 Corset 加入训练组。原 WaterBottle 训练对象及 BoomBox 开发对象的输入、标签和分组保持原身份。独立复核核对全部三个对象的输入和标签字节，并从 Corset 的真实 PNG 像素复算 256 个固定 UV 标签。

Corset 的原始素材采用 CC0-1.0，许可元文档采用 CC-BY-4.0。保留来源、署名和修改说明，许可文件单列标志与商标。此次接收来自上游 main，记录每个原文件的 ETag 和 SHA256，完整上游提交身份未知。 [官方许可](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Corset/LICENSE.md) · [官方元数据](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Corset/metadata.json) · [官方贴图引用](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Corset/glTF/Corset.gltf)

现在有两个训练对象和一个开发对象，满足源码中最少两个训练输入的数量条件。标签表示 glTF 材质参数和均匀 UV 统计。实际视图与 UV 的监督对应、原生图像和三平面特征、盲测以及训练数值仍待验证。两个 PNG、标签和组合绑定留在实验存储。

`prepare.py` 的原请求已退出。验证或复用数据时只读现有文件和小型公开摘要。
