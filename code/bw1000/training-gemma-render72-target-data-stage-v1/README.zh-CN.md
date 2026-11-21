# Gemma render72 缺失的 train-only 部署前提

既有 `tools/control_gemma_render72_native_cpu.py` 的 `bind_map` 要求目标 `training-gemma-usgs-render72-native-CPU-v1/data` 和 `view/train800-swift.jsonl` 已存在；现有本地构建器不部署这些文件。本包只补该安装前提，不重做数据选择、曝光生成、原生编码或训练。

固定新 stage request `0f9b465a7c60419c8350d44e8f2f7dfd`。精确16文件总1,118,352B：原 train72 JSONL、原 data-binding、12张原训练 PNG、原800曝光 JSONL和原 ledger。`INPUTS.json` 逐文件固定源/目标相对路径、bytes/SHA。原binding中保留dev文件/媒体的元数据hash及路径，但本scope不读取或发送dev JSONL、dev PNG、dev/blind gold或blind任何body。没有新增独立样本。

数据仍是已冻结的 USGS 派生图：原train/dev/blind closure不变，12训练图、72独立QA，与800次物理曝光区分。数据来自 [USGS catalog](https://earthquake.usgs.gov/fdsnws/event/1/)，图由本项目绘制，署名USGS数据来源；[版权与署名政策](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)及原输入公版资格随原数据清单引用。不取新的外部数据/官方摄影，不把派生图当官方图或现已通过的新领域效果。

## 本机一次构建与真实回放

固定Python执行 `build_local.py`。它把16个实际已冻结原件按SHA封到 ignored `local-only/training-gemma-render72-target-data-stage-v1-original/payload.private.bin`，再将同一个完整包交给真实 `install(...,local_test=True)` 在全新本地目录安装并重新读取校验16文件。负边界为现存目录拒绝覆盖、变更/越界header在创建目录前拒绝、截断保15/16部分安装。此本地验证不调用目标main，不证明目标数据存在或CPU编码通过。普通本地stdlib工作，未声明本机CPU affinity或内存硬限；包体上限4MiB，准备预算60s。

## 根审核后才允许的目标接口

`receive_once.py --root-go --epoch <root在已有传输前捕获的单一epoch>`，stdin仅上述原包；不内置SSH/登录/下载器、Slurm、模型或GPU接口。根需普通发布并将claim的 `root_GO/data_stage_GO` 置true及刷新SOURCE_FILES；其余训练/Model/GPU/READY保持false。目标main逐源清单核字节、限定HOST POSIX、2CPU affinity和单epoch120s（不重置），SIGALRM约束流接收；不声称内存硬限。只精确创建既有CPU入口期待的一个namespace，存在即拒绝，不覆盖或自动补半包。

包体是 magic+有界JSON header+固定顺序16文件。header必须等于INPUTS；每文件收齐先核bytes/SHA再写独占文件，fsync，全部完成后再读核hash、检查原T02/512数据绑定，输出 bounded receipt。失败保独立intent、已写文件数和原因；UNKNOWN不重播。目标源、stage和CPU任务的身份分别记录，不能延用或重置旧CPU的epoch。

本机未见原CPU的 local dispatch intent，不足以证明远端未执行。原CPU request仍是 `8d6d573cc64b4ab3b082b17682043a11`；认证恢复后根先对账，既有目录/原步骤/未知结果均不得被本包覆盖。未认证、未实际安装、未native512输入/掩码/EOS、未原始初参准入、未设备/真实optimizer/恢复，因此新增READY为0。

controller 的原 `RUN=training-gemma-usgs-render72-native-input-audit-v1` 与本包数据 `TARGET=training-gemma-usgs-render72-native-CPU-v1` 是两个不同目录。实际接收前根须查询原CPU request 的目标 intent、Slurm 标签/步骤及 accounting，保留 UNKNOWN 而不是据本机文件缺失认定没执行。receiver 自校清单只闭合本包内字节；已有普通传输外层还必须固定审核后的公开 SOURCE_FILES SHA 与源提交，并在传输前核对，不把可一起替换的本地清单当独立发布证明。
