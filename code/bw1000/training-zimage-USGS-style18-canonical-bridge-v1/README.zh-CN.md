# T07 本机共同入口只读桥接

`check.py` 只消费本轮新建的 T07 12图 train / 6图 dev CSV 与 PNG。它从冻结 FRAMEWORK.tar.gz 精确取共同 backend 的 `_reject_symlink_components`、`_dataset_resume_evidence`、原 `sha256_path` 及原 step→sample_index 表达式，保留原函数体，执行真实 CSV/文件SHA与路径检查。没有 Torch/模型/像素解码/联网/远端操作。

25个拟定 optimizer slots 对12图产生 `[3,2,2,2,2,2,2,2,2,2,2,2]` 曝光；checkpoint20后真实下一数据序号应为8，20→25尾序号为8/9/10/11/0。此处只验证原代码的索引规则，没有实际 optimizer 调用，也不能证明未来设备的采样器或恢复状态已经通过。dev始终不是训练输入。

原始PNG512²未重新缩放。目标原生 torchvision BILINEAR+center_crop256、文本/VAE、Turbo adapter融合、数值与恢复资格仍欠缺。不存在训练READY增加，不替代原COCO20或其A100参考。

输出只在独立忽略目录 `local-only/training-zimage-USGS-style18-canonical-bridge-v1-original`。目录存在即拒绝重复执行；保留实际结果供根审核与发布，无源图/CSV/提示文本入Git。
