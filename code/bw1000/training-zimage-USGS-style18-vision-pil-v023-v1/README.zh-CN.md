# Z-Image 科学图表原生输入的 PIL 路径复核

使用原 12 张训练图、6 张开发图和各自 CSV。脚本从冻结 DiffSynth 源执行原 CSV 构造、路径路由、图像解码、双线性缩放和居中裁剪。缩放和裁剪调用的是 torchvision v0.23.0 官方源码的 PIL 分支，使用本机 Pillow 13.0.0。此处的 Torch 兼容接口只负责类型分流和日志控制，Tensor 运算分支会报错。没有加载完整 Torch、torchvision 或 DiffSynth 训练环境。 [torchvision 主分派](https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/transforms/functional.py) · [torchvision PIL 实现](https://raw.githubusercontent.com/pytorch/vision/v0.23.0/torchvision/transforms/_functional_pil.py)

18 张实际源图均为 512×512，各执行一次原缩放和裁剪，输出 RGB 256×256。独立对照逐张重算并核对全部像素。原 CSV 顺序和提示文本摘要一致，两个数据分组保持独立，盲测图未读取。只发布摘要，原图、CSV、提示文本及逐图记录留在私有目录。

权重、目标 TorchVision 版本、原生 tokenizer/VAE、实际 optimizer 更新和恢复仍待验证。该结果用于独立科学图表配方，原 T07 正式配置和失败记录保留。
