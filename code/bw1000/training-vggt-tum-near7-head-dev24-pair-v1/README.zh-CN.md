# T12 near7 相机头独立 Dev24 验证

新请求 `58dac28ab5514d5e91ae9109faa84f70` 使用真实完成训练的 `.978` 相机头，在 `1103606/f13r4n09` 上与同一冻结底座配对。原训练只含近景录制，本轮使用目标已封存的 `freiburg1_plant` 与 `freiburg2_dishes` 开发窗口。每臂保留 24 个原窗口，其中 22 个已有合法 RGB 与位姿，另 2 个留在失败分母。因此计划 44 次真实 HIP 预测、0 次训练更新。原 `.608` 两个其他训练头及其结果只作历史证据。

目标安装先核对原 Dev24 stage SHA，以及原 `.978` 的公开终态、原意图、原 worker 回执和训练产物。私有图片、位姿及权重均从目标原目录复用。GPU 端保持原 scorer、同一 224 图像转换、BF16 预测、冻结 aggregator，并校验 near7 archive SHA 和 camera-head 模块 SHA。两臂评分沿用 AUC@5/15/30。

根协调者发布后使用 `launch.ship(publication, snapshot, transport)`，以当前新鲜设备快照和已授权的有限传输完成一次预约和派发。资源为 1 DCU、4 CPU、64 GiB，整个范围 1800 秒，目标控制器 1500 秒，GPU 最多 1200 秒。根协调者核对实际前置、GPU、后置会计和独立零卡清场后消费结果。

在目标使用新安装的独立审核器重算 44 行预测与全 24 窗分母。下列 `heads_sha` 取自本次新目标 stage marker 的 `head_authority_SHA256`。

```bash
label=training-vggt-tum-near7-head-dev24-pair-v1
R=/public/home/aclsylqidf/hygon-inference-refactor-20261014
stage="$R/results/$label"
/usr/local/python3.12/bin/python3 -B "$R/incoming/$label/tools/audit_vggt_tum_cross_camera_dev.py" \
  --inputs "$stage/inputs/inputs-private.json" --inputs-sha 3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6 \
  --heads-sha "$heads_sha" --progress "$stage/score/progress-private.jsonl" --output "$stage/independent-dev24.json"
```

这是已有开发集的独立精度验证。全局盲测、正式 CO3D、A100 性能和真实待派训练仍分别验收。
