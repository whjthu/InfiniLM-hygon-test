# T04 render144 新原生 CPU 资格源码

SOURCE_PREPARED_NATIVE144_CPU_NOT_INVOKED。root_GO/CPU_GO/READ_GO/GPU_GO/Model_GO/training_GO/Dataset_GO/formal17 全 false。原 native150 已合格但不重跑、不继承到新输入；本包暂无 native144 资格，0 stage/native/model/train/GPU/remote/Git/fixture。

复用 training-USGS-T04-wide150-native-CPU-v1 的真实 caller/完整404框架、14件模型metadata、7 Swift与3 wrapper provider pins、完整ENV和原attention/GC。CPU固定 [6, 7]，不动态换mask；原4096/full144、原自然EOS身份及训练非空shifted loss mask检查不变。Torch是与原资格相同的CPU导入；model构造、权重读取、SDK/网络和GPU显式调用均禁止，隐式library映射仍 UNKNOWN，非RSS hardlimit。

原数据 manifest14864B/4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc，仅metadata精确核定。新 INPUT_ONLY_MANIFEST.json 只允许28原payload：24PNG(895438B)+3inputs+trainlabels，总1031596B。两个eval gold只登记原SHA/bytes及read=false/upload=false/required_on_remote=false；完整30原manifest不作远端数据依赖，不上传gold、不为凑30require。source只存chart/group/event哈希，无题目/gold/rawIDs/tensors。

root审/pub后，单次submit将28原payload与小input-only metadata走原普通SSH EOF源码安装；remote按原render producer真实schema id/chart_id/group/kind/event_ids/images/messages/labels:null，完整保留user文本与image，train72按id/chart/group/kind原question精确join训练label并追加assistant `FINAL: answer`；dev36/blind36仅user，不构造或打开gold路径。适配输出3swift+24PNG+小binding留远端私有，不在PC提前decode/tokenize，不生成新题、不变prompt或答案。

同一首300s epoch覆盖本地pub/source/data资格、stage、metadata preflight、原nativehelper、held/tree/post闭合及独立consumer：work220/controller230/held242/tree245/hard248/producerSSH250/reap253；submit已真实接线新进程consume_once，再一次普通SSH只读原namespace metadata/source/provider和已存fingerprints，reader<=35秒且随原295收缩、held reap298、caller299、whole300。无新epoch/重试/fallback/service/timer。失败保144全分母及已完成数；failed producer不会被reader升级。consumer未独立重算tensor math或再encode。

单独新namespace `/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-render144-native-CPU-v1`，max1 request/no replay。source MF与claim普通发布并root明确采用后才允许invoke。入口 submit_once.py --publication <本包publication>。静核 verify_package.py --prepared 只AST/MF/冻结metadata，不运行stage、prepare_native、nativehelper、旧资格或fixture。SOURCE_DIFF.patch为相对已合格parent的Python精确差分，SOURCE_REUSE.json证明processor/template/encode/collator/fingerprint调用和train mask/EOS分支AST相同。CPU资格通过也不构成模型加载/训练/质量/领域晋级或17case验收。assembly v2独立保持GO=false。
