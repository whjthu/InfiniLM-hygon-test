# Z-Image 官方固定revision运行组件清单

只复用root已发布`30e954b9c4010bc91987c3af1d4acdd042bc5a3f`的9290B publisher结果，SHA3ca66ba1…；不改它，不发HTTP/SSH，不写本地权重、不派Slurm。全部GO=false，SCNet原gateway listener当前不存在。本包没有下载器或默认传输fallback。

`RUNTIME_FETCH_MANIFEST.json`精确列19个base运行文件和1个Ostris公开preset，完整URL、固定revision、bytes、SHA、target相对布局、32MiB分段数量。总33018433789B（999个计划range，不是已发请求）。源为Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6及ostris/zimage_turbo_training_adapter@654cd1bf8b3589d9442cb1c5e35221879e1af4f3。

source身份已经闭合：19base中11个是root实际流读正文SHA，8个是LFS size/digest（含tokenizer.json，7个是weight）；adapter另1个LFS匹配。所有运行文件逐bytes/SHA等于冻结A100文件。root此阶段tensor正文读0，本包也0。原77文件完整receipt保留原语义；未纳入运行子集的58项中34项是.source-sha256，24项为其它收据/文档/辅助项，不能把它们冒必需运行组件，也不能把本20项subset SHA冒原77 canonical digest或旧declared model_sha。

已锁581B/65478875…的SCNet stat结果：四个原A100 locator在SCNet为ENOENT，原101槽expected contract已给出。这证明该次四根观察，不是101次逐file body验证，也不证明其它目标位置不存在。未知目标资产路径不猜测，target_asset_ROOT=null；target_provisioned/fullSHA/nativefusion/READY保持false。

## 可执行的清单校验与单段规格

```
python -B manifest.py --repo REPO --manifest RUNTIME_FETCH_MANIFEST.json
python -B manifest.py --repo REPO --manifest RUNTIME_FETCH_MANIFEST.json --asset-id 20 --confirmed-offset 0 --operation-output NEW_PRIVATE_OPERATION.json
```

读取根已发布身份文件并逐项重构比较，篡改URL/revision/bytes/数目即拒绝。第二条只生成一次操作规格，不下载；offset必须来自恢复后目标持久收据并符合32MiB边界，file end生成seal-only。源码只import urllib.parse，没有HTTP或子进程执行代码。首次本机8项检查通过：20身份/总量、11正文+8LFS+1preset口径、main替换和size篡改拒绝、精确首段、末尾seal-only、不齐段offset拒绝、77/19/58/34区别。测试未复制或读取任何weight、COCO20或初始capture。

## 连接恢复后的最短已有路由

现有`tools/stream_internvl_public_segment.py`已用于公开HTTP→内存bytearray（每段最多32MiB）→根显式既有target callback→`stage_internvl_weight_segment.append`目标checkpoint/fsync，最后target完整SHA seal。本机只留JSON收据，receiver临时文件只在目标。不要改用relay_internvl_weight_segment的A100 sender：本任务的源是官方公网，不是私有原机。

旧stream主入口硬绑InternVL shard2/3/4、BASE/revision及TARGET，不可原样用于Z-Image，也不可重提交旧请求。新操作规格已提供必须绑定的asset spec、每文件publisher parent URL/revision、目标相对布局；根的新消费器只复用既有分段与durable receiver primitives，在全新目标namespace接入这些值和当前授权的transport callback。连接失败/UNKNOWN必须先读取同一目标持久offset，不能凭本机日志推测续点或重放。原legacy允许每range最多4次有界网络尝试/单window最多420秒；HTTP206、精确Content-Range/Length和完整SHA门禁不应省略，正常TLS、官方HF域约束保留，不另建隧道/代理/监听服务。

实际consumer的资产绑定和目标assembly应由root审后接入，**本包只交不可变清单及操作规格，未声称已有通用Z-Image传输执行器**。若小metadata端点不提供预期Range，保留真实失败或复用root已核小body缓存，不让200大响应落本机、不放宽权重大小/哈希、不切别的来源。全20文件在目标全文SHA seal前不做模型加载；seal后仍需native tokenizer/VAE、Turbo融合、同输入数值和真实更新，不因源匹配或文件数量晋级。

原COCO20和A100原始input/gradient/initial/optimizer capture全都不在此manifest或迁移范围；继续独立私有保存。SDXL Style18图/训练入口也不与Z-Image权重混用。本轮实际网络/目标写入/optimizer更新/READY均0。
