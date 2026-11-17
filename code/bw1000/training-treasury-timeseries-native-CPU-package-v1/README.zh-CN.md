# Treasury 时间序列原生 CPU 完整执行包

独立新namespace `R/results/training-treasury-timeseries-native-CPU-package-v1`。`native/`是此前已审的6文件逐字节副本，原目录和原CPU结果不改。本包当前未安装、未调用launcher、0新Slurm/原生processor/GPU/训练更新；领域路线仍仅Treasury、VGGT，原17正式用例和原160/108不变。

同一未执行的新CPU scope迁至parent1103606/node f13r1n11，8CPU、16GiB、0GPU。根当前T01账本的1103606分配包含f13r1n10/f13r1n11且added_verified；真实成员及active step仍须临派发核。原330成功环境完全复用：DTK26 sandbox、canonical a68952cb framework、原Swift runtime、MiniCPM Transformers prefix、qwen-vl-utils0.0.14和唯一hyhal系统alias例外。HOST前提首先仅向1087656查询原1087656.330的终态及原intent/outcome/log/sourceMF，再向1103606核当前parent40nodes/320DCU、f13r1n11无active step；本次自己的accounting也只查1103606。旧330只是provider资格，不能作为新432题已过。`native/CLAIM.json`是原6源历史准备快照，字节不改；实际执行资源仅服从本包外层CLAIM及显式launcher，绝不执行历史claim。

HOST实际核393framework、7vision文件、Swift qwen模板d4a35c…、Transformers Qwen3-VL模型f39b36…和9个processor/config/tokenizer元数据的bytes/SHA，全部来自已冻结清单。只复制9个元数据到新model目录，容器不绑定原权重目录；禁止打开safetensors/bin/pt/pth/ckpt。inside记录实际Swift/Transformers/qwen-vl-utils/Torch provider路径及文件SHA，核既有CPU provider前缀、HIP Torch且无GPU初始化或设备可见性。本包不是完整GPU/Native DSO闭包资格。

原数据只供给manifest、48PNG、三个inputs.jsonl和train/labels.jsonl，共53文件/3489522B。安装者依据私有manifest的PNG项选择，禁止额外复制dev/blind labels、windows或selected-source rows；HOST实际调用原native.bind后核真实文件视图恰好53项，多/缺/链接均拒绝。没有原始问题/逐题ID/标签/PNG进Git。目标只读source/data/model；唯一work目录以rw单独绑定，进度写work/native/progress.json，避免整个RUN只读。原6文件中DATA-only/readiness说明仍保留为原资格快照，目标新receipt才决定本次CPU是否过。

launcher在一次入口设置epoch及epoch+600，永久独占launch.lock；失败也不删除，未知不重播。actual srun time按同wall/monotonic余量向下取整为整分钟且≤480，inside≤450并为kill保10秒，client及terminal每个query都缩至同原epoch/自己的窗口，保40秒accounting和最后10秒余量；余量不够不新step。inside另收原deadline和在HOST确定的work_deadline，SIGALRM和timeout均不重置；每样本仍用原checker同deadline。终态helper真实接入；srun超时仅核owned PID/PGID/startticks后清理自己的client，实际Slurm仍必须终态观察，否则UNKNOWN。

目标计划完整432题encode+144训练user-prefix encode=576，原生训练mask及异长collator/meta RoPE；失败保planned144×3和已持久进度，不能缩分母。原图/缩放grid尺寸仅为可读性代理，语义可读性UNKNOWN；实际数字读取、训练数值/更新/恢复、领域效果和正式同A100性能均未验证。即使新CPU通过也无GPU/Model_GO。

本地静态检查只执行新budget实际argv/timeout构造、terminal visibility边界、源码AST/包SHA和6副本一致性。未重复运行旧数据或processor资格。下一根发布后安装source及私有53数据文件，核当前必要条件，再在HOST只调用一次：`/usr/local/python3.12/bin/python3 -B source/launch_once.py --publication <published40hex> --claim-sha256 <newCLAIM64hex>`。这只是明确未来调用接口，本轮未调用。
