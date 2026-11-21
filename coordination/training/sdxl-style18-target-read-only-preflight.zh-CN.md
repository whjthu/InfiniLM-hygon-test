# SDXL style18 目标部署预检

T06 科学图表域使用固定 12 张训练图、6 张独立开发图。部署预检复用已冻结的 portable 入口，只核训练目录和 deployed SDXL 代码。原 T06 `.540` 参数更新门槛失败保留，当前开发数据准备沿新的领域作用域推进。

恢复批准的目标连接后，在同时具有冻结 portable 包和新工具的项目副本执行一次。使用已核路径指定训练目录、部署的 Diffusers 代码根和实际公共训练 backend 文件。

```text
python -B tools/preflight_sdxl_style18_target.py --train-dir <train-only> --source-root <deployed-diffusers> --backend <deployed-sdxl_training.py>
```

若仅观测到实际训练脚本但尚无经独立核对的 Hygon SHA，部署结果为 `DEPLOYED_TRAINER_PIN_PENDING`。结合观察到的 SHA、实际发布记录及部署来源核对，随后可以在同一入口使用 `--expected-hygon-trainer-sha256`。当前脚本缺失、与已核 SHA 不一致或缺少共同检查点参数，均保留为独立状态。A100 原脚本 SHA 仅用于对照，Hygon 使用自身实际部署证据。若训练目录未搬齐，主状态报告缺目录，`deployed_source_state` 仍保留同次部署检查。

只有训练输入字节、目标脚本、公共 backend 与必需 CLI 参数均闭合才报告 `TRAIN_ONLY_DATA_AND_DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING`。此时仍待真实原生 ImageFolder、tokenizer、VAE、固定噪声与初始参数、首步优化器和唯一原设备准入。入口无模型或训练执行，也不触碰 A100 或历史私有传输。候选可登记为待核验，当前没有该范围的 GPU 派发认领。
