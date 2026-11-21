# 训练和推理在途时的 640 卡 busy 采样

原无卡探测在 80 台均因设备计数器权限退出。后来的 640 卡窗口通过了普通 GRES 设备归属检查，适用于全池空闲时。训练或推理在途时，使用已拥有设备的 workload 进程读取同一卡的 `gpu_busy_percent`，避免为读计数器再申请该卡。

协调者从实际 Slurm 父作业、编号步骤、设备归属收据和 PCI 清单冻结当次采样绑定。在同一节点上，每张 PCI 只交给一个步骤。每份绑定记录 `request`、`scheduled_epoch`、`node`、`step`、`boot_id`、`pci`、`mode` 和 `owner_receipt_SHA256`，覆盖其已分配到的 1 至 8 张卡。`mode=workload` 使用现有训练或推理步骤。只有确实由一次正常 GRES 步骤分配给探测进程的空闲卡才使用 `mode=idle_gres_sample`。任一未核验或无权读取的设备保留缺失。

在原步骤内部启动采样子进程，使用同一绑定文件、原 SHA-256 和专用输出路径。采样进程只读取绑定的计数器，持续三个点、间隔两秒。记录在私有目录存档，文件名为绑定文件的 SHA-256 加 `.json`。

```text
python -B tools/sample_training80_owned_busy.py sample <binding.json> <binding-sha256> <report-dir>/<binding-sha256>.json
```

采样计划记录 `request`、`scheduled_epoch`、`ledger_SHA256`、`pci_pins_SHA256`。`bindings` 用绑定文件 SHA-256 映射到其 JSON 对象。协调者独立核对步骤归属、计划及原产物后汇总。

```text
python -B tools/sample_training80_owned_busy.py aggregate <ledger.json> <pci-pins.json> <plan.json> <report-dir> <summary.json> <plan-sha256>
```

只有收齐无重叠的 640 张卡、共 1920 个有效读数时，才输出本窗口全池均值和严格 `>80%` 判定。每组保留应读和实读分母。物理卡覆盖、实际训练更新和持续利用率仍需分别验证。当前仅有本地样例验证。真实作业接入、共同采样和跨节点时钟核对待具备集群访问后执行。
