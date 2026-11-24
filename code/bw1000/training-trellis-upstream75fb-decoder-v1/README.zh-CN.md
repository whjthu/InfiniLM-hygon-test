# TRELLIS.2 固定源码与 CPU 解码器验收

海光直连 GitHub 的 Git 请求原观察超时。使用微软公开提交 `75fbf0183001ed9876c8dbb35de6b68552ee08bd` 的 18 MB 浅 Git 包，保留原 Git 提交、树、浅边界和两份已独立核对的源码字节。普通 fetch 流程要求缺失的历史父提交，导入时先写入已记录的浅边界，再运行 `git bundle unbundle` 和 `git fsck`。源目录不接收检查产物。

框架执行 `3f37d53b1efcea039d2ed808e3b5a1a1916cd596` 中原 CPU 解码入口的解码器构造与严格加载逻辑。两组已落盘的微软权重逐文件从集群盘重算完整 SHA-256，再核对模型的所有键、形状、有限值和原参数精度。几何烟测使用四个解析体素，调用固定 O-Voxel 函数和框架 CPU 查找实现，检查两个三角面的面积。该烟测不产生真实图像生成或留出集分数。

固定父作业 `1103606`、节点 `f13r4n05`、2 CPU、8 GiB、0 GRES。完整启动从一个 270 秒预算计时。进程、源、权重和 Slurm 终态均按实际原件记录。 `SOURCE_PROOF.json` 包含精确输入、源码、框架和原权重证据摘要。

[微软固定版本源码](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd)
