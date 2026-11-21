# 用框架 GLB 做 TRELLIS.2 几何开发集评分

在能够读取原生 GLB 和导出收据的一侧运行 `tools/prepare_trellis_export_prediction.py`。入口复用冻结框架的 GLB 读取实现，核对导出收据自校验摘要、原实现 SHA、文件路径和字节。导出的顶点已经使用 glTF Y-up 坐标。每个对象使用在预测前根据条件输入固定的旋转、米制单位和位移，再套用参考对象的中心与最长边。预测过程中不从该对象的参考网格或预测包围盒拟合变换。

提供两个私有 JSON 文件。调用方须独立证明坐标映射及检查点身份。

- `binding.json` 使用 `TRELLIS_EXPORT_SCORING_BINDING_V1`，包含 `object`、`split`、`input_frame`、`output_frame`、`GLB_SHA256`、`export_receipt_SHA256`、`independent_registration_evidence_SHA256`。坐标名分别为 `native-exported-glTF-Yup-v1` 和 `reference-normalized-glTF-Yup-v1`。
- 配准文件使用 `TRELLIS_FIXED_EXPORT_REGISTRATION_V1`，包含 `object`、`input_frame`、该对象冻结参考 NPZ 的 `reference_SHA256`，以及 `transform`。其中 `rotation` 是 3×3 正旋转矩阵，`translation_m` 是三维米制位移，`meters_per_export_unit` 是正比例。文件内容须匹配绑定中的 SHA。

```text
python -B tools/prepare_trellis_export_prediction.py --glb OUTPUT.glb --receipt OUTPUT.report.json --registration-evidence REGISTRATION.json --reference-root REFERENCE_DIR --binding BINDING.json --binding-sha256 BINDING_SHA --output NEW_PRIVATE_DIR
python -B code/bw1000/training-trellis-public-mesh-quality-input-v1/score_once.py --reference-root REFERENCE_DIR --predictions NEW_PRIVATE_DIR/predictions.json --predictions-SHA256 PREDICTIONS_SHA --split dev --output NEW_PRIVATE_SCORE.json
```

开发分组只有 BoomBox。训练分组的两个对象分别转换后，由调用方合并 manifest 再评分。当前入口为这一独立公开开发配方限制单 GLB 64 MiB、30 万三角面和 2048 像素纹理，范围之外需单独配置和核验。参考及预测各固定 2048 个面积采样点，同一对象的采样种子与检查点无关。

本机检查将 BoomBox 的现有公开几何经原版序列化器写成 GLB，再执行原版读取、已知旋转与缩放、面积采样和现有开发集评分。它使用实际公开网格和测试配准，模型生成、原生模型收据、正式 T10 评分和设备训练仍待实证。
