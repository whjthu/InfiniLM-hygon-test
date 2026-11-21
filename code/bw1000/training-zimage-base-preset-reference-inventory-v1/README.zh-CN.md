# Z-Image native12 后的模型/同输入资产层级盘点

原 `.681` 已12/12原生RGB256与提示通过，`.682`独立post clean，不能重跑它充当训练更新。本包不联系集群、不下载/传输/读取权重、不读原始A100捕获值、不调用Slurm/Git。所有GO=false，READY新增0。

## 已经找到的准确原件

- 原底座：A100 `/mnt/data/nvidia-baseline-20261010/inputs/models/z-image-turbo`，77文件；预置文件：同机 `/mnt/data/nvidia-baseline-20261010/inputs/models/zimage-turbo-training-adapter/zimage_turbo_training_adapter_v1.safetensors`，170128256B/SHA21dc9159…。77+1共33071396964B。完整file SHA已在原CPUv2实际读取验证，但原公开清单明确`official_release_provenance_qualified=false`，不能冒官方release或SCNet已部署。
- 原COCO20：A100 `/mnt/data/nvidia-baseline-20261010/inputs/datasets/coco-val2017` 下20JPEG共3644668B，`metadata-zimage-20.csv`1521B/6513edae…和`manifest-zimage-20.json`1932B/4ddebfd6…。它是正式原用例数据，不用Style18 CSV替代。
- A100首步捕获：`/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-A100-native-first-updates-v1/continuous/first-update-private/progress.json`，2524B/bb9021ac…；原12件捕获共120438246B，噪声1×16×32×32、timestep726、原首步loss0.7768367528915405。原始input/grad/parameter/optimizer值仍在原机，不在本包，也未传输。
- 同一原A100连续25更新合格，但checkpoint20→25恢复比较**失败**：参数maxabs0.00048828125、loss maxabs0.0014133751392364502，原atol1e-6，不放宽。原partial fusion只证明A100同backend路径240目标及新476张量/9922560元素，非独立数值后端。
- 海光并非完全缺attention实证：新增节点f13r2n07的`1103606.27`组件数值PASS、`.28`clean，结果14f75b53…可复用明确范围；不是n14全模型资格。另原n13/a673…UNKNOWN独立保留，不能为拿组件结果重播或洗clean。

已针对现有T07 task/coordination路径字段查找，没有取得SCNet全base/preset/COCO20安装清单。这个范围的“未找到凭证”不等于当前磁盘不存在。不能把以上A100路径直接替换前缀后称为目标副本。

## 最小可执行层级入口

`EXPECTED.json`由上述已发布小清单构建，77base/1preset/22同输入文件/1首步progress共101个槽，重算原model/preset canonical manifest SHA一致。既有`hygon_indexed_model_identity.inspect_listed_files`会完整读取33GB，因此不作为这次轻量盘点调用。

根若知道已存在的目标位置，在新的私有locator JSON中填入：

```
{"base":"EXACT_TARGET_BASE_ROOT","preset":"EXACT_TARGET_PRESET_FILE","same_input":"EXACT_TARGET_COCO20_ROOT","first_capture_progress":null}
```

每组可为null或省略，即UNBOUND；不从A100字段推导目标路径、不glob或扫描HF缓存。根在已授权RO scope中采用root/READ两GO并刷新MF后执行：

```
python3 -I -S -B inspect_assets.py --bindings PRIVATE_TARGET_LOCATORS.json --output NEW_PRIVATE_101_STAT_RESULT.json
```

只lstat/路径规范性/尺寸，不打开任何资产正文，包括模型、图片、CSV或progress。仅处理自身两个小JSON，30秒协作上界由根已有硬有界只读包装收口。结果完整保101分母，权限=UNKNOWN、别名不追随、缺失或未绑定不隐藏；stdout仅各组计数和私有完整结果的bytes/SHA。`SIZE_MATCH_CONTENT_UNVERIFIED`绝不能当完整SHA或融合合格，sidecar也不冒权重内容。

本机6项边界只使用真实冻结101槽和明确空目录fixture：未绑定101保留、不隐式回读A100、空目标101缺失、相对目录/未知组拒绝、0body。未检查任何真实目标模型，也不计数据/native进展。

## 能产生第一更新的最短链

若目标实际资产已存在，先用上面层级1查精确路径和尺寸，再按根新认领做完整body identity（原33GB本机读取不能偷偷带入本scope）；随后目标原生load+BF16 preset fusion240目标/521基座、PEFT476张量/9922560元素/冻结文本与VAE边界。复用原source/runtime及已有attention证据，fresh设备/SDK/PCI准入另核。

正式同输入首步还需原COCO20、精确初值及首步实际噪声/目标等输入的可授权供料/参考口径；目前只有公开manifest指针，不能用seed代替原操作数、不能擅自导出12件捕获。按原层记录prediction/loss/梯度/参数/AdamW及真实更新；原参考不能静默改成Style18。

Style18是独立领域配方，其12个目标图片/提示已PASS，但tokenizer/VAE/同一融合后底座、首步数值和连续恢复仍需自己的实证。其A100同输入参考若尚未形成，应新scope冻结同一输入比较，不能借原COCO20首步loss当它的金标。当前目标locator未给定，不具备可执行GPU首步claim，本包只交付真正下一步的轻量闭合入口。
