# SDXL 完整 pipeline 与已核 UNet 的边界

本轮只读已有公开源码/凭证，不联系SCNet、不下载、不读任何模型/初值正文、不启动Slurm或Git。结论是**尚未找到完整SDXL目标manifest**，不能因已有12张图、20 CLI和UNet而称Style18可训练。

已证实路径来自原真实controller：

`/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-t06-sdxl-unet-login-stage-v1/model`

它被旧container映射为`/model`。原verified-inputs证明UNet 10270077736B/SHA357650fbfb3c7b4d94c1f5fd7664da819ad1ff5a839430484b4ec422d03f710a与config30ebc707…；旧第一步用缓存操作数，可不加载tokenizer/VAE/两个text encoder。因此它不证明七组件根已完整。

原2026-11-10 runtime.json的4个观测候选根为下表。那次全部`unet_bytes=null`，只代表当时，没有当前不存在的证明，也没有对这4处重新查询。

| 记录路径 | 当时证据 |
|---|---|
| `/public/home/aclsylqidf/hygon-inference-refactor-20261014/models/sdxl-1.0` | unet_bytes=null |
| `/public/home/aclsylqidf/hygon-inference-refactor-20261014/training/models/sdxl-1.0` | 同上 |
| `/public/home/aclsylqidf/shared/sdxl-1.0` | 同上 |
| `/public/home/aclsylqidf/shared/models/sdxl-1.0` | 同上 |

对公开results与coordination/training JSON检索`tokenizer_2`/`text_encoder_2`未取得可用完整清单；不是整盘库存证明。原A100 DEPENDENCIES的 `/mnt/data/.../models/sdxl-1.0` 是另一端目录，不能冒目标路径。2026-11-22新Style18实际target-preflight仍明确native_tokenizer_VAE_verified=false，也没有weights manifest。

新`inspect_pipeline.py`提供根协调者可直接在已授权只读scope内执行的具体入口：

```
python3 -I -S -B inspect_pipeline.py --root EXACT_EXISTING_SDXL_ROOT --output NEW_PRIVATE_METADATA_RESULT.json
```

可以显式重复`--root`，最多5处，无默认搜索、glob、HuggingFace缓存扫描或fallback。整个调用配置/词表正文合计最多8MiB、单文件2MiB；固定12个metadata slots（model_index、scheduler、两组tokenizer_config/vocab/merges、两个textencoder config、VAE/UNet config）。只导出bytes/SHA和类名，不导出词表正文。四组件权重仅lstat固定标准文件/索引名，不打开bin/safetensors或索引正文；**即使索引存在，也不证明shard齐全**。别名/权限失败明确UNKNOWN/待审，不追随未知来源；损坏或变化metadata不会PASS。真实模型全body SHA与native输入资格始终false。调用时限/2CPU等由根现有RO包装限定，本模块不创建进程/服务/SDK。

本机只验证了新入口AST、空显式目录12+4分母保留、相对根拒绝、0正文与0READY，使用明确空目录fixture，未把fixture冒真实模型资格。没有重跑Style18/native/旧终态或读取被拒初始LoRA。

下一最短解除动作：根在一个新只读scope里核已知候选根当前元数据，若现存完整pipeline则固定实际根+metadata SHA、对应既有模型body身份/版本，再做原生双tokenizer/VAE与数值入口；若只有UNet，则先保留缺六组件的实际结果，不能给Style18发GPU任务，也不能把“有模型同名目录”当权重齐全。本轮新增READY=0，所有GO=false。
