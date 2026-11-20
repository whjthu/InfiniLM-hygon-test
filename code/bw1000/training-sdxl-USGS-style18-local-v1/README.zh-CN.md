# SDXL 科学图表风格媒体门禁

这是一项独立新域配方的数据前提，不替换 T06 的 COCO64 正式配方。复用已冻结 USGS 数值派生的图像，12 张训练、6 张开发；原 6 张盲图和全部问答标签不读取。图片由项目本地绘制，不能称为 USGS 官方图或照片。

`once.py` 固定请求、独占目录、单 epoch 60 秒，复用已实测 Windows Job 的 2 CPU/2 GiB commit 限制。`prepare.py` 对 18 原 PNG 全解码，转 RGB、BICUBIC 缩至 256×256，重读输出逐像素核验，输出 ImageFolder `file_name/text` 格式 `metadata.jsonl`。`audit.py` 独立核输出图像、元数据和原分组。原训练/开发/盲测的组与事件隔离原样保留；实际读取仅 train/dev 媒体及 manifest 元数据。

文本是本项目固定编写的通用视觉风格描述，不是原问答答案、COCO caption 或 USGS 标注。18 图共用一个描述，因而只适合小型版式风格工程实验，不能验证数值条件生成、精确文字或领域能力。256 像素文字可读性尚未人工资格验证；缩放后的像素身份均独立冻结。训练时仍需同一冻结噪声/timestep、可训练参数和 A100/海光输入副本，以及原生 tokenizer/VAE/实际 optimizer 更新与恢复检查。

来源依据：

- [USGS 数据版权与署名](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)：USGS 生产数据属于美国公共领域；官网第三方图片并非一概开放。本包不取官网照片、人物或标志。
- [USGS 署名要求](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted)：保留 `Credit: U.S. Geological Survey`。本包沿用原 ComCat 公开数值身份和本地绘图来源说明，不追加原始数值数据的独占权利声明。
- [Diffusers SDXL 官方输入示例](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py)：ImageFolder 支持图像及 `metadata.jsonl`。这只是上游接口依据；本项目冻结共同框架 SDXL backend 的具体接入与 native 训练仍待核实，不能继承上游 main 或 VLM 资格。

实际本机结果在 `local-only/training-sdxl-USGS-style18-local-v1-original`，原件不进 Git。首次普通读取原 manifest 的 PermissionError 保留在工具记录，随后按既有授权精确只读成功。未使用网络、Torch、模型、权重、GPU 或集群通道。海光原 SDXL 参数更新误差 FAIL 保留；新训练 READY 增量为 0。

下一步由 root 审阅本包与真实结果，再将此 ImageFolder 数据视图绑定现有统一 SDXL backend 的数据参数；必要原生 CPU 接口/噪声身份资格通过及目标资源恢复后才考虑一次独立配方的真实训练。不得重跑 `once.py` 覆盖目录；重复调用会在独占 mkdir 处拒绝。
