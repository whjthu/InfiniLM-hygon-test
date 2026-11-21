# SF3D 材质头效果验收

正式验收入口 `verify_training_acceptance.py` 从实际私有预测文件重算三项 MAE。A100、海光训练后和海光训练前三组共用同一批 RGBA、查询位置、标签和评分器。每组预测按实际训练检查点核对。三项指标必须引用同一批头输出。文件留在项目私有存储中，公开清单保存可核对的 SHA-256。

每项指标的 `sample_plan` 包含同序的 `material_RGBA_artifacts`、`material_query_artifacts` 和 `material_references`，每项带 `sample_id`、`status` 和现有 `artifact` 引用。原查询位置按小端 float32 的 XYZ 三元组保存，文件长度须与 `query_points_per_sample` 一致。选择原训练配方和留出计划实际使用的查询位置与个数。

材质记录是独立 JSON，字段为 `sample_id`、`status`、`RGBA_SHA256`、`query_SHA256`、`checkpoint_SHA256` 和 `material`。参考记录的检查点字段为 `null`。参考反照率是 RGB 三元组，原生预测反照率逐查询保存 RGB。粗糙度和金属度分别保存单元素数组。每样本先对全部通道与查询点取绝对误差均值，再对样本等权平均。失败保留在原计划及分项计数中，正式 MAE 门槛要求三臂完成全部预测。

`material_policy` 为 `sf3d/native-material-head/sample-equal-mae-v1`，`scorer_SHA256` 绑定实际 `tools/training_material_mae.py`。固定视角 LPIPS 和原模型的目标运行仍需实际证据。
