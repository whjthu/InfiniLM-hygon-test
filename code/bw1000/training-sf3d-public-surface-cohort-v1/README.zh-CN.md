# SF3D 表面积材质训练集与统一入口

用 `complete_once.py acquire` 取得发布方的四个 Corset 素材。入口限制目标主机、文件名、字节数和 SHA。Corset 素材采用 CC0-1.0。 [发布方许可](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Corset/LICENSE.md)

用 `complete_once.py derive` 从原网格的 18,324 个三角面生成固定的 4,096 个表面采样点，沿用同一 PBR 解码器和标签字段。原 WaterBottle 训练标签和 BoomBox 开发标签按原 SHA 复用。结果绑定两个训练对象、一个开发对象及 12,288 个采样标签。开发标签不参与训练身份。

用 `render_native.py <新目录>` 生成私有训练清单及共同框架覆盖层。标准 SF3D 和旧 atlas 的入口仍可选择。新 `sf3d_surface` 路由核对固定顺序、两张真实输入图和两份标签，并把训练输入身份绑定到恢复校验。已有的原场景编码、损失和优化器调用来自同一冻结后端。

本机验证调用了真实共同 CLI 路由和新输入加载器，成功读取两个对象的 PNG 和材质标签，随后在 CUDA 入口停住。拒绝了开发集混入、伪造标签摘要、改变训练身份、同时选择两种数据配方及调换输入顺序。真实设备特征、优化器训练、盲测效果和正式 T11 验收待执行。
