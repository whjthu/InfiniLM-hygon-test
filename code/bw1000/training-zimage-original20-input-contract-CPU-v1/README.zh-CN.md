# T07 原COCO20输入合同与恢复取样审核

本包只审原T07冻结数据指纹和确实存在的本地原生数据源码，不生成替代caption或图片，不把SDXL ImageFolder配方混入Z-Image。

实际输入身份来自 `training-zimage-A100-native-first-updates-v1/BINDING_IDENTITY.json`（120160B、a3d62f6cc61ba8d472d61dc7394c262fa167187d577c6c27f28a7bf3930f3ae1），并核对该包SOURCE_FILES登记。原20图共3644668B，各图body SHA及各prompt SHA都唯一，行序无重复、无绝对路径或目录穿越；CSV1521B/6513edae65017e75494e9b49d644f8f4d46877cddf1c8acf2ddfb33c83f2816e，manifest1932B/4ddebfd6c9aa4fc37caef3add589d470c1c54e3a20f0ed26092d37b4352b4e4c。这里只核原已公开指纹，不重新宣称实际媒体版权/署名或留出划分资格。

原生完整源码确在 `training-zimage-target-native-meta-CPU-v1/upstream/diffsynth/core/data/`：`unified_dataset.py` 4972B/b6e0729fbd00deee8b0d03dd463df43ce7e653ecea3f75c18741d4cc03ac027f；`operators.py` 7735B/c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4。与原A100 binding和目标186 provider身份相同。CSV经pandas读，图像键为image，其他字段保留；原T07监督文本字段为prompt，不能换成SDXL的text。原生处理为RGB、torchvision BILINEAR resize和center_crop到256，并非SDXL的BICUBIC派生图。

`check.py` 已实际执行原 `UnifiedDataset.__getitem__/__len__` 两方法的未改AST，对真实20项指纹记录做元数据路由；为避免引入Torch，只去除类的Torch父类且不调用构造器、媒体operator。并执行共同backend `run_zimage_training` 中未改 `sample_index` AST。得到25项序列=20原项+前5重复；每项曝光次数为前5项2次、后15项1次；checkpoint10/20/25的数据游标分别10/0/5，20→25五项与连续尾段相同。它不是25次真实优化器更新，也不是CSV解析/PIL变换/原生训练资格。

精确本地缺口：已查该原A100包的 `original-inputs.stdout.private.json` 及对应 `local-only/training-zimage-A100-native-first-updates-v1/original-inputs.stdout.private.json`，均不存在；不能据此声称全电脑无副本。原CSV、manifest和20JPEG已记录在A100 `/mnt/data/nvidia-baseline-20261010/inputs/datasets/coco-val2017`，本轮没有访问该机或重取原件。若root另有合法本地副本，仅需对上述CSV/manifest和原binding逐图pin核，再用真实原生operator验证；不能凭指纹重建提示词或替换媒体。

目标186只是meta/PEFT资格，预置Turbo融合值、目标实际完整模型输入、HIP整模型首步数值、25步/新进程恢复仍未闭合。源码读取和元数据路由不重跑186或旧恢复，不解决这些缺口。全部GO保持false，训练READY增量0。无HTTP/SSH、模型/权重、GPU、Git修改或共享status变更。
