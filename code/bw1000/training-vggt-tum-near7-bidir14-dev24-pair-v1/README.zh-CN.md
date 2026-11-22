# VGGT 双向 near7 训练头 Dev24 独立验证

同一冻结底座与实际完成 25 步的 `1103606.1078` 相机头分别预测原 Dev24 两个录制中的 22 个可用窗口，每臂保留 24 个窗口。两个原缺失窗口以及任何无效或退化预测留在失败分母。计划 44 次 HIP 预测、0 次训练更新。沿用 .1038 的输入、224 像素处理、BF16、评分器及 AUC@5/15/30 口径。

目标先绑定原 .1077/.1078/.1079 全部会计、原输出和 .1081 清场。新头的实际文件长度为 864,719,552 字节。目标预测器在加载前核对完整 archive SHA 和装入后的模块 SHA。局部两对留出已有的 AUC@30 提升只作工程证据。

根协调者发布且取到新鲜 `1103606/f13r4n09` 设备准入后，用 `launch.ship(publication, snapshot, transport)`。分配 1 DCU、4 CPU、64 GiB，整个范围 1800 秒，控制器 1500 秒，GPU 1200 秒。当前源包的 `GPU_GO=false`，由根发布时启用，`training_READY=false`。

实际前置、GPU、后置会计和更晚的独立零卡清场齐备后，使用目标已安装的复核入口。其评分重建比较沿用 .1038 已实测的 AUC 浮点舍入边界，只对 AUC 字段容许 2e-15 绝对误差，预测事件、结构、原分母及每行几何复算依原审核器核对。

```bash
label=training-vggt-tum-near7-bidir14-dev24-pair-v1
R=/public/home/aclsylqidf/hygon-inference-refactor-20261014
/usr/local/python3.12/bin/python3 -B "$R/incoming/$label/audit_terminal.py" "$R/results/$label" \
  --actual-accounting "$ACTUAL_ACCOUNTING" --independent-post "$INDEPENDENT_POST" --output "$AUDIT_OUTPUT"
```

仅完整独立审核后应用预先公布的规则。若 AUC@30 低于同轮冻结机基线、AUC@15 低于同轮基线的 95%，或失败数增加，则停用 near-only 增强路线。证据缺失时保持未判定。正式 CO3D 和效果验收仍依对应的独立用例。
