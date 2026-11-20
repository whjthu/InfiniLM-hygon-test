# 接入冻结共同框架的 SDXL 数据配置

已找到完整原件：`training-framework-bd24ec3-v1/FRAMEWORK.tar.gz`，发布 `80f5355cf79d67c4ec1678d92efc8c02016924dd`。精确成员 `framework/generation/src/gencuda/backends/sdxl_training.py` 为 11576B、SHA256 `1965014be6e0040ffb6ba033f491569c5530a8a8d665012b46247a5ff43ed5e7`，与 a68952c 共同393文件清单逐SHA相同。相应 genmlu backend 亦在归档，19707B、SHA256 `65814732b3d2eb4bec29198c2594bbfbd0e84ceb364592ac542a23c133c11ffb`。这是本地已发布源码，不是当前 Diffusers main，也不是新下载。

实际函数为 `run_sdxl_training(weights, config, output_dir, source_root)`。它接受 `train_data_dir`，向 `source_root/examples/text_to_image/train_text_to_image_lora_sdxl.py` 传 `--train_data_dir`；拒绝旧 `dataset_root/dataset_metadata/max_steps`。`adapter.build_config()` 精确绑定上一轮实际18图的4个小元数据pin，只将12图训练目录传入新域配置；6图开发留出，盲图/问答金标不读。`run_via_common_backend()` 通过实际 provider 整文件SHA再调用该共同入口，未另造训练循环。

本机 `check_entry_CPU.py` 已实际调用冻结原函数AST，验证完整25步/256/batch1/accum1/rank4/BF16/seed20261121/checkpoints20,25参数接线；在唯一 `subprocess.run` 处主动截停，原生训练子进程0。另一次真实入口调用确认旧配置键仍拒绝。空模型/空训练脚本仅是测试路径存在性哨兵，不是真实模型或官方训练脚本；测试不读取图像、权重、Torch、tokenizer或VAE，也不重跑原18图准备和审计。

有一个必须保留的真实语义：共同 backend 固定 `lr_scheduler=constant`、`lr_warmup_steps=0`；`config.warmup_steps=5` 只排除前5步性能记录，不是5步学习率预热。本新配方直接沿这一入口，不冒改正式T06。LoRA alpha、双编码器及VAE原生处理仍要由实际已部署训练脚本/运行资格证明，CPU接线不能代替。

下一最短缺口：本地归档没有完整 `train_text_to_image_lora_sdxl.py`。已存在的冻结补丁 `generation/patches/diffusers-sdxl-training-evidence.patch` 仅是补丁，不能当完整脚本执行。因此本结果只证明项目共同入口到真实数据目录的配置接口。root 若持有已部署的官方训练脚本身份，应核其固定 source root/文件SHA以及图像和text列；其后才能做原生CPU输入、固定噪声/timestep、初始参数与一次真实优化器更新资格。不能将上游示例版本或元数据可解析冒为 native tokenizer/VAE合格。

原T06海光数值更新FAIL、目标不可达/资源未核和新训练READY=0保持。新公开结果不含媒体、caption正文或事件ID；所有实际配置和测试路径留 ignored `local-only/training-sdxl-USGS-style18-canonical-adapter-v1-original`。不提交Git、不SSH、不GPU。
