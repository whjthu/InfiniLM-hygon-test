# T07 新领域图像 CSV 数据前提

本包把既有 `training-USGS-multievent-render144-v2` 的原始 512×512 PNG 分流为 Z-Image 的 `image,prompt` CSV。训练12图、开发6图，原盲测6图不读取；事件与地理/时间 closure 分组沿用原 manifest，不更改划分或依据模型输出挑图。唯一请求 ID `181839cf0cb74a93b96225a94c9c8273`，已有目录即拒绝重做。

这是一份独立科学报告视觉样式 recipe，使用本地撰写的共用描述，不读取 QA/gold，也不宣称训练出数字/文字生成能力。18张图是已有数据在另一模型入口的复用，不是18张新增原始数据。原 T07 的 COCO20、性能参考、数值问题、融合/模型/设备资格均保留，未被此包替换。

## 与真正入口的连接

已按原 T07 source identity 核对 DiffSynth `unified_dataset.py` 4972B / `b6e0729f...` 与 `operators.py` 7735B / `c02a73eb...`。共同框架 `generation/src/gencuda/backends/zimage_training.py` 31044B / `6526be51...` 的实际调用使用 CSV、`data_file_keys=('image',)`、`UnifiedDataset.default_image_operator(height=256,width=256,height_division_factor=16,width_division_factor=16)`。

实际本机步骤：复制原 PNG bytes（不使用 SDXL 的 BICUBIC 派生图），完整 CSV roundtrip，逐图执行原始且未改写的 `ToAbsolutePath`、`LoadImage`（RGB）与 `UnifiedDataset.__getitem__/__len__` 函数体，以及原 `get_height_width`。这些 AST 函数直接操作真实新文件，只省去顶层 Torch/pandas/torchvision imports；不冒完整 DiffSynth 包导入资格。CSV 由标准库读取，未执行原 pandas load_metadata。

训练时原生链会做 BILINEAR resize 和 center_crop256。本机只验证原始 RGB512 和原生目标尺寸，**没有执行 torchvision 缩放、tokenizer、VAE 或模型**。不以 Pillow 的其它插值替代原生资格。18图逐字节保留，因此将来目标CPU必须真实核原缩放/文本与融合后的模型入口，再进训练。

生成私有 `training-data-overlay.json` 只有 `dataset_root/dataset_metadata/dataset_manifest/dataset_sha` 四个新数据字段；train root 与 dev 完全分流，dev不进入训练。目录是未来目标位置声明，不证明已部署。其它模型、预置Turbo adapter及融合后底座、rank4、目标层、AdamW、预热、精度、checkpoint/recovery均由原配方和后续目标准入独立验证。

## 来源与使用边界

原数值来自 [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/)，图像由本项目基于固定数值本地渲染，不能称 USGS 官方图片。署名为数据来源 U.S. Geological Survey；不使用其商标背书。

[USGS copyrights and credits](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits) 于2026-11-21由 primary web 文档核实：USGS 自产数据/信息属于美国公有领域；第三方图片不自动适用。本包没有下载其摄影/第三方图片。派生图、固定输入/分组/原body SHA 由已冻结 manifest 关联；raw、caption、事件清单和图均仅放忽略的 local-only。

`once.py` 单次60s（含源码读取和启动），实际 Windows Job 限2CPU/2GiB commit，复用已合格 helper SHA `c52244b5...`，不安装/联网、不 SSH/Slurm/GPU、不读权重。文件与失败分母保留。成功仅代表本地 T07 CSV 和真实原图加载前提，READY、training/Model/GPU/formal17 全为 false。
