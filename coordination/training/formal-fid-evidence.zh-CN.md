# SDXL 与 Z-Image 正式 FID 证据

统一验收入口从固定的真实图像和三个检查点对应的 2048 维 Inception `pool_3` 特征独立计算 FID。协方差采用样本分母。真实图像数可以与生成图像数不同。数学定义沿用原论文，实际特征提取器、权重、预处理和样本须按配方冻结。 [FID 原论文](https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)

在原 `FID` 样本计划中增加 `fid_reference_image_SHA256`，存放真实图像 ID 到图像内容 SHA 的映射。保留三个检查点原生成样本计划和逐项结果，成功记录的 `artifact.SHA256` 使用生成图像内容 SHA。缺失或失败的记录保留在原分母中，整组 FID 暂不能用于正式验收。

在原 `FID` 样本清单加入 `fid_inputs`，引用一个带 `path` 和 `SHA256` 的私有 JSON。该 JSON 包含和原清单一致的 `case`、`framework_commit`、`dataset_SHA256`、`scorer_SHA256`、`metric`、`sample_plan_SHA256`、`checkpoint_SHA256`，以及下列内容。

- `policy` 使用 `FID/pool_3/unbiased-sample-covariance-v1`。
- `feature_extractor` 指向 SHA 等于 `scorer_SHA256` 的 JSON。该 JSON 包含同一 `policy`、`layer=pool_3`、`dimension=2048`、`implementation_SHA256`、`weights_SHA256`、`preprocess_SHA256`。收集阶段还须保留实际执行的提取器和环境证据。
- `reference` 对象和 `arms` 的 `A100`、`Hygon`、`Hygon_before` 每项均有 `sample_ids`、`image_SHA256`、`features`。真实图像按 ID 字典序排，生成图像按原计划排。`features` 引用矩阵格式的私有 `.npy` 文件，行对应图像，数据类型为 float32 或 float64。每份矩阵支持 2 到 50,000 行。

通过 `python -B tools/verify_training_acceptance.py .` 核验。确认四份特征文件在当前项目可读的私有存储中。Git 只提交源码、小型结果和清单，原图和特征矩阵留在实验存储。核验器可审计输入身份和距离计算，实际图像生成及 Inception 特征提取仍需要原运行证据。
