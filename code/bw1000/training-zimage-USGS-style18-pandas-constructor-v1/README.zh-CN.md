# T07 真实 CSV constructor 补充证据

此前style18包验证了真实图像、CSV标准库roundtrip与原路由，但没有执行 DiffSynth constructor 的 pandas 读入链。本包只补这段：原 `UnifiedDataset.__init__`、`load_metadata`、`__getitem__`、`__len__` 函数体逐字保留，实际读取已冻结train12/dev6两CSV。验证 pandas read_csv/iloc.to_dict 后的所有image/prompt值、类型、顺序和复制语义。数据分流不变，dev不进训练；不读PNG、QA/gold或blind正文，不重做下载、录制、图像预处理或曝光生成。

为避免引入未安装的Torch，测试容器省去了torch.utils.data.Dataset基类，直接执行四个原函数和原unused cache operator初始化；metadata_path固定为CSV、data_file_keys为空，不走pickle或图像operator。源码身份为原统一T07的unified_dataset4972B/b6e0729f…及operators7735B/c02a73eb…。这是真实pandas输入函数证据，**不是完整DiffSynth包/native训练环境导入资格**。

本机确有pandas3.0.1；没有torch/torchvision。所测范围没有使用Torch stub模拟模型/缩放，不继承目标pandas版本资格。保留真正缺口：原共同入口要求的T07 launch wrapper SHA f0bc54e4…原字节尚未在本机取得；BILINEAR+center_crop256、tokenizer/VAE、预置Turbo adapter融合、初始可训练状态、实际设备/数值/25步恢复均未由本包验证。

`check.py` 使用独占忽略目录 `local-only/training-zimage-USGS-style18-pandas-constructor-v1-original`。本轮只运行一次，完成18/18 metadata行，新增图像body读取0、encode0、GPU/优化器0。此CSV包含本地撰写的视觉样式描述，不是USGS/COCO人工caption，也不证明地震数值图生成能力。已有图基于[USGS公开数值及署名规则](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)派生；数据身份和train/dev/blind分组沿用style18原件。

READY、训练、Model、GPU GO保持false。根持有普通发布和目标派发；此本机结果不改变原T07失败及正式17项参考。
