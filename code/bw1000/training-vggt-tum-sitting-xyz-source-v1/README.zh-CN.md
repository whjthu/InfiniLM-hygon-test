# 新录制 T12 训练数据前提

仅获取 `freiburg3_sitting_xyz` 的一个真实 RGB-D 对，不训练、不加载模型。
先读取固定 Voxel51 revision `f21e73c5bf0d91fc66f3c5759c9d438e52ab24a6` 的 MCAP LFS identity/footer/summary，再取 TUM 官方归档前 8MiB 与小位姿文本；原 RGB PNG 与至多两个 MCAP depth chunk 均按范围有界读取。非 206 的大对象响应在 body 前拒绝，无重试或整包回退。唯一目录存在即拒绝。

排除原训练 `structure_texture_near`、`nostructure_texture_far`；开发 `cabinet`、`teddy`、`freiburg1_plant`、`freiburg2_dishes`；盲测 `long_office_household`。只读既有 manifest/规则身份，不打开开发/盲测像素或标签。录制不同不等于物理房间独立：Freiburg3 可能共享办公室。此新录制有动态人物，不提供动态掩码，后续必须保持原几何门槛，不据像素挑样或降低门槛。

许可与来源：TUM 为原作者；Voxel51 发布者声明原数据及其转换为 CC BY 4.0，并明确 depth 为原 16-bit PNG、RGB 重编码 H264。本 scope 仅使用官方原 RGB；镜像只提供原样 depth，不把 H264 当原 PNG。须保留 Sturm、Engelhard、Endres、Burgard、Cremers (IROS 2012) 与转换来源署名。

- 原发布方：[TUM 数据主页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)、[官方下载目录](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)。目录的旧域名链接公开重定向至 `webshare.cvg.cit.tum.de/g/rgbd/dataset/`。
- 转换发布者：[Voxel51 dataset card](https://huggingface.co/datasets/Voxel51/TUM-RGBD#parsing-decisions)。CC BY 4.0及原样深度说明于 2026-11-21 通过 primary web 文档核实；不声称本轮取得了官方完整归档。

本轮 web 工具点击 TGZ/groundtruth 链接未获得正文，属于 web 访问失败；本程序的真实 HTTP attempts、状态、字节、EOF 和私有 SHA 单独记录，不混同两种观察。

`acquire_once.py` 复用既有成功采样器的逐字节副本 `frozen_pair_acquisition.py`：时间差 0.1–0.5s，深度/位姿最近差不大于 20ms，第一合格时间对，平移不小于 5mm。实际数据源获取成功仍须独立原 `tum_vggt_pair.prepare_pair` 几何门禁、媒体去重及统一模型输入资格；READY、训练、模型效果均为 false。完整1对/失败分母保留，不补位或换录制。

运行：固定本机 Python 执行本目录 `acquire_once.py`。原始 HTTP、PNG、时间索引、位姿只写忽略的 `local-only/training-vggt-tum-sitting-xyz-source-v1-original`。时间预算180s、最多20个请求、body累计18MiB；网络速率不保证成功。没有 Slurm/SSH/GPU/权重操作。
