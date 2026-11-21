# T07 真实18图：官方 PIL 分支缩放/中心裁剪参考

此包补此前明确缺失的 **512×512→256×256 图像变换本机前置**，没有再做 CSV constructor 或采集 USGS 数据。18张既有真实来源图保持12 train/6 dev；原盲组、gold、模型答案均未打开。真实本机 resize18次、center_crop18次全部完成，失败0、missing0，0.313秒。生成18份只用于目标数值对照的 RGB256 参考 PNG，共404474字节；不能替换原 formal/SFT 输入，不能声明目标已部署。

## 真实源和许可

原 DiffSynth `ImageCropAndResize` 类来自已经冻结的 `operators.py` 7735B / `c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4`。其类/方法体不改，仍使用实际 `max` 比例、Python `round`、BILINEAR 和 center_crop。

本机没有 Torch/TorchVision。此次单独取得 [PyTorch 官方 vision v0.24.0 functional.py](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/transforms/functional.py)、[官方 PIL 后端](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/transforms/_functional_pil.py) 和 [BSD 3-Clause 许可证](https://github.com/pytorch/vision/blob/v0.24.0/LICENSE)，总81496B。全部保留真实 body SHA、ETag、HTTP200、长度、自然 EOF。tag 并非本次目标已部署版本的证明；目标版本/源正文对应仍未知。官方正文和许可证原件只在 local-only，不复制进 Git。

原输入来自 USGS 公有领域数值的本项目派生图。[原始数值入口](https://earthquake.usgs.gov/fdsnws/event/1/)及[USGS 使用边界](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)沿原已核来源许可；不是 USGS 官方图片，不使用其商标背书。未下载图片或任何权重。

## 可审阅的执行边界

`pil_branch.build(source_root, operators)` 返回 `transform, trace, proof, invoke`。`transform(real_RGB_PIL_image)` 调用原 DiffSynth 类，再到官方源码的 PIL 分支。

这里是**显式 source projection**，不是完整 TorchVision 包导入。没有伪造 torch 模块，也没有安装/修改共享环境：

- 保留官方8个 `_functional_pil` 函数的完整 body；只移除 `torch.jit.unused` 注册 decorator。
- 官方 functional 去掉5个仅记录 API usage 的 JIT/logging 块；在严格真实 PIL 输入门禁下，把4个 `isinstance(img, torch.Tensor)` 谓词特化为 false。其它计算/尺寸/枚举/插值映射/裁剪体均由原 AST 执行。旧 Tensor 路由保持不可到达，不支持 Tensor，不填 Tensor 替身。
- 原 DiffSynth 方法使用一个显式 PIL 调用命名空间；它不是 `sys.modules` 中的 torchvision 包。原 `.transforms.functional` 和 InterpolationMode 调用路径被投影到上述官方分支，未手写 resize/crop 算法。原方法 body AST 不改。
- 每图实际 trace 均为512→256 resize、256→256 center_crop；实际二维 PIL 分支不经过 tensor antialias 实现。非 PIL ndarray 输入拒绝测试已实际通过。

因此结果只能说明本机这条官方源数学分支及既有数据可执行，**不能表述为目标原生 TorchVision/完整 Dataset/tokenizer/VAE 资格通过**。完整 source/执行 AST 对应都在 SOURCE_PROOF 和私有实际结果中，可逐项复核。

## 身份、隔离和视觉检查

逐图原 PNG 全 SHA、原 RGB512 像素 SHA、输出 RGB256 像素 SHA、PNG SHA、尺寸和分流均实核。训练3来源组/72事件，开发2组/36事件，train/dev 的组、事件及输出像素 hash 交叉0；未据答案或变换结果选样。

私有 `RGB256-reference-binding-private.json` 是将来目标 **同一原512输入** 的像素参考，不是新增训练输入或缓存资格。实际目标 CPU 应执行其真正安装的完整 native operator，再比同 ordinal/source pin 的 RGB256；版本或像素不符如实记录，不能把本机参考偷偷当目标输出。

另实际只看 train/000.png 和 dev/000.png：全图、双图、轴、类别、数值、页脚未见裁切/遮挡。两张粗视觉检查不能证明 OCR 或模型生成效果，其余16张只程序核尺寸/像素。

## 两次 source 操作与停止边界

第一次默认沙箱 request `3bb488f111fa4c6d9dd558f15821e83e`：1次 urlopen 尝试、0响应/0body、WinError10013，0.047秒，已知失败。没有发生自动审核拒绝，原件保留。

根明确授权受审批的新只读公网操作后，request `c7d3a9b946f74937bd6987749f4f785f` 经 require_escalated 审核通过，独占新目录，3 GET全部完成，3.640秒。它不覆写第一次失败，不使用集群/代理/镜像，也不是未知执行重播。原 `.py` 源下载器未改；新 wrapper 硬核第一次失败 result 的 bytes/SHA 后才调用已冻结下载逻辑。

本机输入审计 request `6dac1ccbbd7747a7a227b15d2862f78a` 另独占目录，0 GET。源码读取/图像处理采用本机逐输入45秒检查；没有冒 OS 硬超时、affinity 或 RSS 保证。三个 once 原目录均已存在，不得重跑。

新增 READY=0。仍缺：目标真实 provider/版本与256像素、原 launch/底座/预置 Turbo adapter/融合身份、text/VAE、初始参数、真实反向/optimizer 首步及连续25/恢复5、正式性能与效果。所有原 FAIL/UNKNOWN 与原 T07 COCO20/A100 配方保持独立。无 SSH、Slurm、GPU 或 Git 操作。
