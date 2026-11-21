# SF3D 原生材质输出导出

对同一冻结留出计划分别导出 A100、海光训练后和海光训练前。实际执行使用统一框架的 SF3D 后端、原始两份底座文件和同一个已核验的原生包。执行需要一个获准的加速卡，原生包一律使用离线资产。权重、查询位置和原始输出仅留在实验存储。

```text
python -B tools/export_sf3d_native_material_effect.py \
  --plan <绝对路径> --plan-sha256 <原计划摘要> --output <新绝对路径> \
  --framework-root <统一框架目录> --sf3d-source <原生源码目录> --weights <原始底座目录>
```

私有计划记录 `case=T11.material-head`、`arm`、数据集摘要 `dataset_SHA256`、原生包摘要 `provider_sources_SHA256`、样本查询数 `query_points_per_sample` 和有序 `samples`。每项样本带 `sample_id`、`RGBA` 和 `queries`，两个输入分别用 `path`、`bytes`、`SHA256` 绑定。查询位置采用小端 float32 XYZ。样本输入及查询顺序沿原冻结计划，后端直接消费已校验的原始字节。

`checkpoint` 的 `kind=full-native-checkpoint` 时绑定原文件 `path`、`bytes`、`SHA256`、`global_step`。执行时核完整检查点、原底座、原生源码、步数和 36 个材质头张量身份，只载入材质头参数。海光训练前使用正式训练已绑定的初始检查点。若该检查点即原始底座，可用 `kind=native-base`、底座 SHA-256 和 `global_step=0`。

每条成功记录保存逐查询的反照率和两个标量材质头，失败保存同 ID 的状态。输出 `manifest.json` 同时绑定计划、检查点、实际设备、后端与完整计数，可将其 `records` 接入 [统一材质验收](training-sf3d-material-effect.zh-CN.md)。原生模型执行和固定视角 LPIPS 均待实测。本机的导出测试使用夹具预测并验证文件可由同一验收读取。
