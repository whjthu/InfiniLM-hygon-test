# VGGT dev24 空档验证

现有 VGGT 单卡控制器在父作业 `1087656` 的 `f11r4n19` 上比较 base 和两个已完成的训练头，处理 24 个固定开发窗口，单次控制器上限 1500 秒。该验证产生 AUC@5/15/30 等效果证据，计划优化器更新为 0，训练储备计数为 0。

恢复已认证登录后，在同一目标工作区使用 `tools/preflight_vggt_tum_dev24_backfill.py` 查询既有原评分范围。入口只读 Slurm 记账、已有 staging 原件和固定控制器清单。应保留返回的 `state` 和原始 JSON。已有评分标记或任意精确标签时，对账该评分原请求。若已有 staging 目录缺完成收据，对账旧 staging，保持原命名空间。只有 `TARGET_STAGING_REQUIRED` 表示本次没看到旧 staging 目录或评分标记，根可用现有 `stage_vggt_tum_cross_camera_dev.py` 按原已封存私有包准备一次新的 staging。

`ORIGINAL_STAGING_VERIFIED_DEVICE_AND_HOLDS_PENDING` 提供 `stage_SHA256`、`manifest_SHA256` 和 `head_authority_SHA256`。它绑定 2 个训练头、22 个可用输入对和 24 个计划窗口。读的是小型原元数据及源码，实际模型和检查点内容由已有验证控制器在运行时核验。

随后沿既有设备 collector 和派发交接重新核对 `f11r4n19` 的本机原保留、PCI、KFD、PID 与 Slurm 步骤。该节点目前还受 `vggt-tum-eight-pair-training` 和 `vggt-tum-continuous25-dev3` 两个未决原请求保留。两个原请求均需实际对账。现存评估控制器需要根绑定唯一评分请求与普通发布 SHA，使用原目标 staging 路径及返回的 `--stage-sha`、`--manifest-sha`，统一派发前完成对应认领。工具输出 `device_admitted=false` 和 `approved_for_dispatch=false`，不提交 Slurm 步骤。
