# T03/HIP 新USGS72完整输入CPU入口（未执行）

固定请求 `8110b01cd24846f8b6fcd08f0abda5a0`，独立namespace `training-internvl-USGS72-native-CPU-v1`。只在根审核、普通发布并采纳后，通过既有普通运输运行一次；本包当前root/CPU/READ/GPU/Model/training GO全部false，无目标安装、Torch/processor/设备执行。

## 真正数据与入口

14个DATA文件来自已发布 `661dcec94d0c3fcc333458a00e7c883b9f089ed3` 的T03平台输入资产：72个独立训练QA、12原PNG、train JSONL及binding，共506,060B。未增加独立数据，未带dev/blind正文或权重/初始参数。`DATA_CONTRACT.json`逐文件核bytes/SHA，`INPUT_ROWS.json`只含72个同序ID/完整行hash和目标图像路径，无原问题/答案。

复用已真实通过的originalCPU360/v5所用普通容器前端、系统hyhal别名只读bind、原393共同框架、13模型元数据、原HF basename、目标数学provider合同、meta504/43,646,976参数和原语言GC生命周期。原worker硬编码64/旧输入值reference未带入或重调。保留原wrapper链和原生`internvl`/`internvl_hf`/template_type=None/SDPA/GC+。

worker使用两个原生训练模板：max_length=1,000,000和原4096。每个原样本各encode一次，字段指纹完全相同才允许collator一次；计划144次encode、72次collator。完整问题/图像和FINAL监督保持，不追加EOS、不手动缩短/替换模板。实际监督EOS、prompt前缀-100、未掩码label==input IDs、完整attention、finite CPU输入字段和原生selected-logits语义均为实际门禁。full/bounded两个模板独立核真实3层encode包装链；内部retry策略沿原3且实际retry次数仍UNKNOWN，调用计数指外层API attempted/completed。

进度在原生调用前后落盘，保72个planned/completed/attempted/failed/unattempted行以及144/72调用分母；失败不补位，不能因最后一行失败丢掉已经消耗的调用。无旧64值指纹复用，无A100144新输入资格自动继承。meta只证明结构/source，不证明初始权重或数值等价。

## 原预算和清理链

整个300秒epoch从本地源/Git/DATA打包前开始，包含上传、HOST核验和一次Slurm；actual Slurm上限240秒并按原余量整分钟缩水，native work最多epoch+210，原45秒资源/清理余量不增。源与DATA单次共同上传，目标目录存在即拒绝，不覆盖未知半包。默认继承8CPU、12GiB、0GRES；step cgroup charged-memory 8GiB为采样soft gate，不是RSS或硬限。来源中的old node/parent只是冻结候选，实际HOST仍必须核分配所有者、成员和该node无编号步骤。

保留v5 qualified held/tree/owned/RSS/cgroup握手、std-lib READY→HOST memory gate→release、绝对Apptainer、无`--rocm`、三GPU可见性mask=-1；不查询CUDA可用性/卡数。weight文件与显式device/network open guard保留，隐式library/SDK查询UNKNOWN。正常退出也必须有原step实际C0、两层held/tree/post证据；SSH255/timeout/UNKNOWN不重播、不释放或宣称GPU准入。

## 实际调用与独立消费

根采纳后在新隔离Python进程中加载本包，调用 `launch_once.main(pub, mf_sha, transport_start=root_callback)`。没有默认helper、SSH、ProxyCommand或运输fallback；裸CLI/缺callback在任何Git/打包/跨端动作前直接拒绝。回调前仍核源的普通Git blob与dirty范围、原661d DATA合同blob、独占原本机intent与原时限。根回调必须仅使用当前明确获准的原反向通道，迅速返回已审二进制stdin/stdout/stderr Popen句柄；包内继续发送body、在原295秒停止并在298秒前有界回收。context只含固定8110请求、原epoch/deadline、固定目标Python命令、source manifest/目标scope/非秘密hash，不含凭据。回调异常/失响应/未得可审句柄即UNKNOWN、不重试；stdout/err仍私有。独立reader不改为任何新运输入口。

Slurm accounting现在严格解析所有非空7列，不能丢弃格式错误行后认为原label不存在；必须恰有当前parent行，原label若存在必须绑定parent.NUM与指定node。`srun_invoked=true`只表示bounded srun命令调用尝试，不是分配证明；只有实际编号行才计actual_CPU_steps=1，未见行就是None/UNKNOWN。独立reader只有实际1/C0及全清理证据才可PASS。

`read_terminal.py`及`TERMINAL_READ_CLAIM.json`已预置。终态后根传真实publication/request/epoch/sourceMF/claim和新的只读epoch，只读原metadata：72行身份、144/72真实计数、完整mask/EOS/selected-logits、providers、meta结构和原accounting/资源清理。它不重编码、读图/原13metadata/393框架或查询新GPU；当前异host登录PID观察保UNKNOWN，不冒software失败或GPU可释放。失败原reason仅private；公版只hash/类型/phase。

本地仅新的身份/计数边界fixture及Python3.10 AST/清单/打包上限检查。没有目标PASS、实际worker、性能、质量或训练READY；原模型权重、initial/RNG、GPU数值和真实25+恢复仍需独立准入。数据沿原[USGS来源与署名](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)，本地图非USGS官方图；原17配方和既有FAIL/UNKNOWN保留。
