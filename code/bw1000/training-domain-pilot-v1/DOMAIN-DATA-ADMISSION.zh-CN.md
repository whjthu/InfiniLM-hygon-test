# 通用图文对照去重与室内导航候选

## 通用 LoRA 对照去重

`check_generic_holdout.py` 按原始 `code/bw1000/training-vlm-original-assets-v1/original-bindings.json` 的 64 个排序条目核对海光原训练 JSONL、各图像的规范路径、文件大小和实际 SHA-256。原图全部属于 `coco-val2017/images`。两个目标留出库存仍缺可核对的精确路径、内容 SHA 和独立冻结后的唯一图像数，因此尚无实际交集结果。

```text
python check_generic_holdout.py --original-bindings <原始绑定文件> --coco-cider-inventory <现地清单> --coco-cider-sha256 <冻结SHA> --coco-cider-images <唯一图片数> --textvqa-inventory <现地清单> --textvqa-sha256 <冻结SHA> --textvqa-images <唯一图片数>
```

库存由各指标的既有输入原件在其授权存储端构造。根对象只含 `metric` 和 `images`，其中 `metric` 为 `coco_cider` 或 `textvqa`，每个唯一图像条目只含 `source_image_ids` 列表及 `image_SHA256`。COCO 规范 ID 格式为 `coco:<无前导零的十进制ID>`。TextVQA 同一图的多个题合并到一个图像条目，同时保留原生来源 ID 及已知跨库别名。按既有留出清单冻结实例时还要核对问题或字幕分母，图片列表只用来检测训练输入泄漏。工具仅读现地图片和库存，只输出命中数与摘要，检出交集退出码为 1。它能发现同来源 ID 和完全一致的字节，图片重编码后的近重复需要已有来源别名或另行审核。两份库存与留出定义对齐后再解释零交集。缺原件时继续保持已有权限和状态，不申请评分输出外传。

本机 CPU 验证已读到真实原身份清单的 64 个零基排序条目和 64 个唯一 SHA。合成小样本验证了 ID 与 SHA 各自命中、同一张图片并集只计一次、TextVQA 的 COCO 别名、零交集、SHA 或分母失配、伪造 ID、修改原训练媒体。未读取海光私有留出内容或执行 GPU。

## TUM RGB-D 室内机器人相机位姿

选用 TUM RGB-D 官网发布的 RGB 图像和与相机对时的运动捕捉轨迹。发布方现行页面将数据许可标为 CC BY 4.0（明示例外除外）。保留 Sturm、Engelhard、Endres、Burgard、Cremers 的作者归属，标注 RGB 裁剪、姿态坐标变换和派生标签。首批范围为办公室、机器人设备区和家庭/办公室场景。可公开下载的 mocap 轨迹直接做可追溯标签。[发布方与许可](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) · [官方序列](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) · [原始论文与测量方法](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf)

真实可达性核对只成功获取 TUM 官方 `freiburg3_long_office_household` 的位姿文本。解析响应后 UTF-8 内容为 592,911 字节，含 8,710 条时间戳记录，SHA-256 `e255d5d9c5bab915132c7ac5ec4e23f9f1efe98f765b0c9b26ac74e40b02533f`。这一步核对了记录格式和行数，图像/深度包、其他候选的标注和训练设备仍待验证。[官方位姿文本](https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household-groundtruth.txt)

先按物理环境和完整采集序列分组。可使用 `freiburg1_room` 一类办公室序列作为训练候选，`freiburg2_pioneer_slam` 机器人设备区做开发候选，`freiburg3_long_office_household` 做盲评候选。下载帧之前核对场景关系和来源证明。同室多次路线、变体、相邻帧始终进同一划分。盘点时保留序列、RGB 时间戳和原图 SHA，按帧内容 SHA 查跨划分交集，并按时间留出采样间隔。若两个候选物理场景有交叉，将它们并入同一划分并启用另一独立场景。只采有 mocap 的窗口，从同一序列选两帧并验证时间最近匹配；避开长期缺少真值的轨迹区间。留出组应逐步扩充到多个独立物理场景再作能力外推。

接入 T12 时增加独立的原生数据适配器。对 RGB 用现有 224 像素入口，同时传播裁剪和缩放对内参的变换，按帧产生 `[1,2,3,224,224]` 输入和对应内外参。先用一条授权的小图批次与原相机头的尺度、坐标轴、四元数顺序和 camera/world 方向做 CPU 重投影验证，确定同步误差、两帧实际平移和可视重叠。随后单 DCU 实测原 VGGT-1B 在线冻结 aggregator 的 camera-head 一个真实更新，记录 loss、梯度和参数变化。建议为这次资格范围先留 4 CPU 与 1 DCU，预处理本身 0 DCU。后续短训及和同预算通用相机头对照另认领；按采集序列做 AUC@5/15/30 和失败分母，并可追加 TUM 原生 RPE/ATE。独立领域试验保留原 CO3D 任务身份和 acceptance 状态。
