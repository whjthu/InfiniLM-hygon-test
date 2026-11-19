# T04 海光原64条 native CPU 与 meta504 查证

独立 training-llava-target-native64-meta-CPU-v4，根协调者批准一次 CPU 派发及其终态独审。GPU、训练和正式验收资格均未发放。未重跑A100或USGS CPU，未改T03 CPU360 v5、任何原framework或旧请求。未发现同scope已invoke资格；仅在本包正式发布且目标节点无冲突后调用一次。

复用CPU360 v5已过完整300epoch、8CPU/12GiB/0GRES、worker210/Slurm240/cleanup45、namespace/no-replay、Apptainer CPU readonly runtime/systemalias、numeric-step cgroup v1/v2 charged-memory原生释放前gate、held/tree/UID/post/accounting/预注册独立terminal consumer。固定1087656/a14r3n14（T04账本节点），不扩资源，不建service/poller。reader源与原控制gate保留，只换T04语义；source/MF及freshparent/节点/label admission实际过后才有一次Slurm资格。

已知海光metadata-v2/llava/model14件；COCO64 hip JSONL26817B及64图片，真正原插件与common393都使用原路径并在compute独立observer新核SHA。HOST06:01:49UTC14SHA/COCO64/dispatcher原SHA点时记录是准入线索，不能冒已完成compute资格。四shard均缺不阻断此次CPU meta，不读取/索取任何shard/initial正文。原baseline metadata/modeling Python源码是14件的一部分，包含14个metadata-only新modelview；没有权重软/硬链接或下载。

T04实际原A100入口对应custom AutoConfig/eager compatibility/verified dynamic class，完整meta构造并由原Swift prepare_adapter+PEFT测504 trainable/43646976/name-shape-dtype、visual及aligner冻结、语言/视觉GCfalse，全部参数meta。原512图/4096上下文、model_type/template llava_onevision1_5，完整原64 native encode/collator；lossmask和原EOS身份实测，不补EOS、不降context、不执行post_encode或embeddingforward。原actual custom forward不支持logits_to_keep，原Swift policy必须False，所有完整labels/logitrows保留，不沿用InternVL选择逻辑。

actual target decorator三层/所有provider路径/源SHA独立测量。custom dynamic modeling.py须等于原14metadata SHA；底层HF/Swift/PEFT平台源码记录来源和coherence，不伪装A100源码数值资格。原A100 pre-model64 encoded及collated fingerprint分别逐项比较并单列bool；结构/native语义PASS与bit诊断不同，不能据位相同或post-model不同证明领域/数值质量。原200条post-model FWD-prehook fingerprints不读、不套为CPU前置目标。

保存小摘要/SHA与完整失败64分母，无rawIDs/encoded tensors/weights进Git。隐式库device查询仍UNKNOWN，显式GPU初始化/设备FD/权重打开被拒；0fullTrainer、0实际模型值、0前后向/更新。Model/GPU/training/effect/formal17全部保持false。独立reader只消费本次context/数值step C0/held/tree/cgroup/post与原生元数据，不重新处理图文或构造meta，不重读14metadata/weights。

入口launch_once.py --publication <本包pub> --source-manifest-sha <本包MF>。verify_package.py只stdlib AST/MF/metadata closure，不运行fixture/Torch/旧资格。源码可审阅不等于实际CPU PASS；若真实target依赖或constructor不兼容则完整FAIL并报确切阶段，不猜后端/回退/重跑。

本版复用原始 64 条输入和 meta504 验证。CPU375 的导入失败已封存。本版按冻结的 Transformers 5.5.0 源码校验，在模型导入期间临时添加上游 5.12.1 已有的缓存类别名，退出导入时撤销。读者必须验证实际 OneVision 模块绑定及恢复回执。

CPU376 终止于别名执行前。此版本将原模型的文件路径传给已有 SHA 校验函数。本机检查执行了 worker 内实际校验表达式，覆盖匹配文件和替换文件。

CPU377 已越过缓存导入和结构检查。本版只读挂载海光已有的 qwen-vl-utils 0.0.14，校验七个文件、实际导入路径及包版本。
