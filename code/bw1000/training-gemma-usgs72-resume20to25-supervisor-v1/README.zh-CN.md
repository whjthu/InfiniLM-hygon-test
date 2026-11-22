Gemma USGS72：独立原生 checkpoint20 → 25 恢复5更新

前提是真实连续 .851：1103606.851 COMPLETED0/183秒/8DCU、pre850/post852 C0、独立reader853 C0。上游ASSET/continuous-independent-terminal.json固定7528B/SHA ba9e7917eb04b298de2104aa5fce80f38b19aa39486d18a2f1f681f446c07a09；原操作outcome SHA e033cf6ca72c8d7cd4ad73e39b11b463a4d98ab384cb59d4ce407a29b581bfbe、native-outcome SHA34b621bffcb505620ef45b68acf3be778fc555face98d558038ef9a64dbdfeb1。仅根此前实际消费证明，不由本机合成测试冒原件独审。任何pin/完整25更新、8rank/100micro、accounting和cleanup资格不符都在新GPU前拒绝。

新操作namespace：R/results/training-gemma-usgs72-resume20to25-supervisor-v1。
新queue/transport request：66e39b4b1aeb4c129646b5c0db3c33cb；不可变CORE logical TRAINING_REQUEST仍247fa3682a114542ac3640fbc726aaae。与实际连续的派发ID/label/操作目录全部不同。新独立原始epoch<=2400秒，不伪称沿用已结束连续scope的原epoch；未延长或重放旧作业。不可变CORE worker接受新scope的<=2400时限并重新核上述旧checkpoint gate。整个配对真实更新计划仍25+5，不是再跑25。

CORE仍为R/incoming/training-gemma-usgs72-fresh-initial-domain25-v1-source，pub16081fc/MF d3e8a643…；ASSET仍R/results/training-gemma-usgs72-fresh-initial-domain25-v1。完整原生resumed-plan、396源码、Gemma16权重、train72/12图和800重复曝光、.845 staging全部复用，不下载、不重stage、不再编码资格。新输出仅ASSET/resumed/train，首次不存在才可启动。容器额外将ASSET/continuous、原independent gate和原操作目录只读挂载，不能覆盖连续记录。原A100 initial.pt从不读；这仍是独立target-fresh领域实验，不是原正式17复现。

根的最短步骤：采用本包root/GPU/training/READ执行位并重封MF→普通发布→通过现有已授权transport回调调用launch.launch(transport_start=ROOT_CALLBACK,publication=本包commit,local_output=新私有独占目录)。无默认SSH/helper/服务。launch复用已实机修复的ROOT.as_posix+绝对POSIX/精确label校验，先source+intent落盘再-I-S明确src入口。实际训练仍必须新的1103606/f13r1n12无owner准入、fresh<=45秒8PCI/KFD/完整HOST同UID FD/空端口29932。旧清场不是新准入。

父HOST keeper、七列accounting、squeue逐行membership、单GPU Slurm调用、Bash显式$0、原held/tree/RSS逻辑直接沿用成功v2。whole2400/native1980/GPU Slurm最多2100/worker清理2080，同一新epoch按剩余缩短，保独立post/accounting。RSS256GiB是采样soft stop，Slurm请求224000MiB；VmRSS缺失UNKNOWN而非0，峰是下界。原生max_steps仍25，从checkpoint20实际恢复参数、optimizer、scheduler、全部8rank RNG和cursor后，只执行21..25五更新。native loader100→skip80→余20micro/rank，合计160次forward，仍bs1×8rank×accum4；不是增加160独立样本。只保存新checkpoint25的14个必需原生成员。

不改任何注意力、优化器、模板、精度、seed、sampler或恢复数学；真正进入CORE/worker.py resumed和原GemmaResumeWitness。8rank恢复-before-RNG、native-data-skip、一次RNG load、20输入与loss、5实际更新state都要求完整。旧连续25、恢复5的真实分母各自保留；失败/partial/UNKNOWN不重播。无dev/blind读或模型效果评分。

已实际接入6564B原tools/compare_gemma_domain_recovery.py，SHA4db332a8865816514a057bca411c2e5e9a52373b81ee610a664d785b39e21402，副本逐byte相同。它比较8rank恢复前step20 adapter/optimizer/scheduler与连续20、每rank恢复RNG、原native cursor、全部160尾部输入和loss（逐值/指纹exact，无放宽阈值）、40个rank update的三类state，以及最终25的per-rank RNG。live state和原生checkpoint完整SHA身份均保留；不把相同序列化文件SHA冒独立typedstate反序列化资格。

若原生5更新完成但数值比较FAIL，native完成/实际5更新与清场仍分别记录，exact_paired_recovery=false，不能称恢复合格。物理资源清场release与数值恢复PASS分开，避免一次有诊断价值的失败被洗成0更新或占住已清资源。controller返回engineering字段，最终read_terminal的passed才是严格恢复比较结论。

GPU实际终态后独立post原PCI/boot/KFD/端口、held与strong owner退出均需C0。预置read_terminal.py在原计算挂载f13r1n12上0GRES/2CPU/4GiB/45秒运行，参数与连续reader相同（--publication/--epoch/--source-manifest-sha）。它核当前原件/完整5分母/原checkpoint25身份，再独立重新运行原comparer，对比producer pair-comparison的完整对象和evidence SHA。输出≤16KiB摘要，完整比较私有留ROOT，既不重model/GPUquery也不改原continuous gate。login与compute st_dev差异不忽略。

本机仅新3项纯stdlib fixture：严格原25 gate负例；exact比较器8rank/160/40成功、一项loss不符仍保全部分母、缺rank保planned分母；新resumed调用/只读旧continuous挂载及真正-I-S隔离source导入。fixture不是实际恢复。旧3/9/17测试未重跑。全部GO初始false、目标调用0、更新0、READY0。Model/formal17/effect/performance仍false，根审/发布后才唯一实际执行。
