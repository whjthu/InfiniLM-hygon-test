Gemma 同主机8卡首forward配对机制诊断（新scope、默认0执行）

原851连续25更新、860恢复5更新及169项数值失败都不改。本包不重投旧作业、不覆原输出、不读旧A100初态，也不把新诊断叫领域/性能/正式17资格。唯一数据仍为已资格USGS train72 QA/12图/800重复曝光，dev/blind不进入任务；不下载或重新stage。

新dispatch：48217c9d9ef84b2c88cff01a4cd025ea；原logical训练cohort247fa3682a114542ac3640fbc726aaae仅作为只读参考。新PAIR_ROOT=R/results/training-gemma-usgs72-first-forward-paired-diag-v1，输出untraced/train、traced/train，目录必须从未存在。原7528B/ba9e7917… independent gate、outcomee033cf6c…、native34b621bf…与checkpoint20十四pins全部硬绑定。原ASSET、原continuous和原操作目录在GPU容器只读。

根采用/普通发布后，以既有明确授权transport callback调用launch.launch，原POSIX receiver/独占意图/未知不重播规则复用成功v2。一个2400秒原始epoch，外部pre0GRES→单个8DCU/64CPU/224000MiB租约→独立post0GRES/accounting。native工作stop1980，cleanup2080，Slurm最多2100，剩余留清场/原账。SOURCE初始所有GO=false，无默认SSH或服务。

GPU步骤内的lease_keeper在HOST运行，不导入Torch/SDK；依次启动两个独立Apptainer进程，每个进程才加载真实模型并从同checkpoint20原生恢复至25（各5更新、各8rank×20forward）。不是同进程重新set_rng/复制state，也不复用A的模型。每个进程结束必须held/tree和inner实际8rank回执完整，第二臂开始前再次做完整HOST同UID FD/KFD/8PCI/boot/空端口fresh检查；不降为容器内partial FD检查。外部pre的物理8PCI/boot也必须一致。父进程不持GPU上下文，前臂GPU holder不为0就拒后臂。为避免监听器TIME_WAIT混淆，A固定29932、B固定29933，仅改变rendezvous端口；pre/post核两个端口，native HIP PCI映射还要与每臂fresh HOST gate相符。

不改原396文件。native_entry逐当前源/模型/数据身份和actual800 SHA核验后，从原resumed-plan派生仅新output、附加observer external plugin与对应port。原checkpoint20、模型目录、dataset、512上下文、BF16、SDPA、GC=false、LoRA/冻结、AdamW/cosine/累积/seed及25终点不变。diagnostic_plugin继承原188e0781… GemmaDomainResume：先且只先调用原super on_train_begin/end，原完整adapter/optimizer/scheduler/RNG恢复、100→skip80→20、step21..25门禁仍执行。基类方法无替换、无数学补丁。

“no-trace”精确定义为不启用e950ac8d的完整light tracer；两臂都加同一最小common_rng_tap身份witness，覆盖全部8rank首4个forward的entry/exit。它只读哈希RNG和少量DDP标志，不消耗/恢复RNG、不读模型payload、不改返回值。若没有这个共同witness，只能知道load端点一致，不能声称真正首forward RNG一致。本包不隐瞒这个共同观测开销。B额外启用e950已发布轻trace，仅rank0及control rank1的4个forward，记录完整dtype/layout/scale/context的私有摘要；A的enabled=false路径返回None，完整tracer不触碰model/backend。完整trace原source字节不变。

pair_compare必须核两臂各8rank/20输入/5更新/原生cursor/RNG、独立PID-startticks、真实参数变化、所有160输入/loss对、40 state对、8恢复state及RNG/final RNG，并检查64个共同首4 entry/exit RNG对。只有实际匹配后才same_independent_input_rng_qualified；不是从原load端点推断。observed_noninterference_exact另要求loss/state也全等。失败/缺rank仍保160/40/64计划分母。两臂各14成员checkpoint25的稳定整SHA和cursor也留在新私有scope。

这是一组固定顺序A→B的机制诊断，不是随机化因果试验。内核/缓存/allocator时间条件可能随顺序变化；即使B与A相同也只证明这对运行的观测一致，不证明修复原860。若不同，只能再结合共同RNG和B的context定位，不能直接宣称trace导致或解决误差。原851没有首forward trace，无法事后造补。

终态consume预置read_terminal.py：根在同f13r1n12计算挂载0GRES/2CPU/4GiB/45秒读取原metadata，以实际publication/epoch/MF为参数。它独立重做配对比较、核两臂源回执/trace文件pin、原始三步C0、全HOSTpost/强owner和各新checkpoint25身份，输出<=16KiB；不重新query GPU或读取PT tensor。资源clean release与diagnostic_passed分开。两个native各5更新，最多合计10实际更新，属于独立诊断，不晋级READY或正式训练验收。原860 recovery_qualified保持false。

本机新5项pure-stdlib：基础plan除output/plugin/port逐项不变；实际subclass函数体在fake base/backend上只super调用一次且完整tracer只有B；160/40/64配对成功和loss/RNG负例；每臂fresh HOST坏字段/过期拒绝；两臂新进程真正-I-S隔离导入。不是实际Torch/HIP/model验证。首次3通过/2ERROR保留：POSIX比较在Windows误用str已改为as_posix，另一项是fixture缺trainer.model；最终5通过。旧套件不重跑，原文件未改，目标实际调用0。

最快独立真实资产候选（非Gemma、非InternVL/VGGT）：T09原794 checkpoint10→25的15更新恢复回归，复用原单DCU已实测底座/cache/Muon/provider；不是重复已完成828的20→25。checkpoint10已公开pin7479771B/4ce48e5e…，common native循环对completed_global_step通用，最短缺口是新15分母/steps11..25 comparator和根fresh核cp10/原权重tuple/资源；不需下载大权重。当前仍非READY，不能把旧C0当新scope准入，也不碰原fixed-first-replay holds。
