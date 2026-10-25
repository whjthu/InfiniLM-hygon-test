# Kimi K3 投机推理：质量优先的高速验证路线（r9046）

截至 2026-10-26 01:24 北京时间，建议继续现有 DFlash2 span8 草稿，接回原始高速 W8 target verifier；停止将 KDA 和通信强制改成 canonical T1。705 的组合虽修复了定位用的 536/537 差异，却使组件时延 1.739→6.143ms，停用理由是慢。用户已明确允许 token 不同：初期基本合理正确，随后以任务 benchmark 无明显掉点判断，旧 exact-token 拒绝不再作为质量判定。

## 已有实测与真正缺口

- 997：现有真实 target 五层 features + 已训练 BF16 draft，4 个上下文 512/520/528/536，greedy matched prefix=5/1/1/2。warm 重算约19.5–20.2ms，包含 context projection、5层、head 和 selector；不含 features 捕获、传输、文件读取和加载。它不是在线接受率或端到端加速。加载约20秒，GPU peak10,033,227,264B。源文件 `Kimi-actual-terminal-r9028.json`，remote result SHA651be3349cc66c321e907f0702ec55680d60ec9d8ac2972665641bb9c8f3632c。
- 675：原W8全模型四块约319/295/294/294ms；只因后续一处greedy token差异被断言停用，32进程正常关闭。这个verifier可重新用质量测试评估。原FAILED/audit1不可改写；新试验用新handle。它还未覆盖partial rejection/replay/EOS。
- 678/705 表明不同执行形状的数值关联改变；不能据少量ULP差异推断语义质量失败。705仅组件，无整模型质量或服务加速结论。
- 目前没有完成当前DFlash2+高速W8的在线 draft→verify→commit/replay→emit cycle。应优先补这个闭环。

## 算法语义

标准 speculative sampling 保持目标条件分布，不要求与另一次相同seed的AR样本逐token相同。候选y由实际q采样，验收概率min(1,p(y)/q(y))；首拒绝处从normalize(max(p-q,0))采样；全接受时从额外target row采样bonus。temperature/top-p/top-k/penalties必须在p和q各自定义中一致、且q与实际proposal采样方式一致。[原论文](https://proceedings.mlr.press/v202/leviathan23a.html)

本轮先greedy：从本次W8 target logits逐位计算argmax，接受与候选相符的最长前缀，再发出本次target的首个纠正token。这里的相符是接受算法本身，不是要求整个生成序列==旧AR。浮点形状差异使本次p和T1路径不同，因此实际工程质量靠benchmark，不伪称相对旧T1位级无损。作者参考代码同时给出了greedy和拒绝采样实现。[DFlash参考代码](https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py)

DFlash2的selector会按前驱token重打分。若后续做sampling，q必须是selector在实际前驱下的条件分布，不能代入raw LM-head softmax，也不能先确定性argmax再假称从softmax采样。第一版只做greedy，保留该扩展接口。[作者selector训练](https://raw.githubusercontent.com/lightseekorg/TorchSpec/main/torchspec/models/dflash2.py)

## 方案比较与草稿选择

| 方法 | 可借鉴内容 | 本项目选择 |
|---|---|---|
| EAGLE3 | 多层feature融合、训练时模拟后续预测；公开最高6.5x只适用于论文模型与设备 | 需要Kimi特定训练权重，未核实到优于已有draft的现成匹配权重；不盲换 |
| DFlash | 一次并行产生整块；论文Qwen block16平均接受约6–8含bonus，有超过6x案例 | 可借其并行draft/rollback设计；不能搬其倍率 |
| DFlash2 | 并行backbone+轻量局部卷积+前驱selector；避免独立位置不协调 | 首选现有已驻留/可复用的7.386GB Kimi专用权重 |
| MTP | 若模型自带训练好的next-token模块，可低成本复用 | 官方Kimi config `num_nextn_predict_layers=0`；没有现成head可直接开关启用 |
| Medusa | 多head+tree验证/typical acceptance | 无现成Kimi head；树分叉需KDA分支状态，集成成本高；typical acceptance是有损路线，需任务质量评测 |

来源：[EAGLE3论文](https://arxiv.org/abs/2503.01840)、[DFlash论文](https://arxiv.org/abs/2602.06036)、[DFlash2作者报告](https://inco.ai/blog/dflash2/)、[Medusa作者项目](https://github.com/FasterDecoding/Medusa)、[Kimi官方config](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json)、[MTP官方实例](https://github.com/deepseek-ai/DeepSeek-V3)。这些公开方法的数值不是BW1000实测。

首选 `lightseekorg/kimi-k3-dflash2` revision `e77935fb4804e17eb55085bffd045eae1d779769`，文件7385765464B，LFS SHA1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2。5层、H7168、taps[19,37,66,78,90]、4SWA+1Full、span8=anchor1+proposal7。它是针对官方Kimi训练的第三方草稿，不是Moonshot官方发布。embedding全字节匹配及head来源已校验，完整训练target revision/量化/AttnRes feature stream仍UNKNOWN，不能据此宣称完全兼容。[草稿发布者模型卡](https://huggingface.co/lightseekorg/kimi-k3-dflash2)

备选 `modal-labs/Kimi-K3-DFlash` / `z-lab/Kimi-K3-DFlash`：同目标的已公开DFlash，作者提供block8与16，并在B300上报告GSM8K接受长度5.905→7.984、HumanEval6.013→8.358。它确实给出更长块的训练来源，但不是现有DFlash2改个配置即可得到；本轮不下载新大权重。[发布者模型卡](https://huggingface.co/modal-labs/Kimi-K3-DFlash)

## 混合状态与最小执行闭环

1. target继续原TP16PP2，n08..11共32卡，Math0，D/E canonical修补OFF；复用678的taps-enabled native，而不是705单层DSO。pair API active/shadow/state_pool17已存在。
2. draft仅n01单卡：997 peak约10.03GB，申请c8/host32GB/1DCU；原权重远端复用。预热在测量前完成。
3. prefill输出首个anchor，捕获五taps；512×5×7168×2=36,700,160B。后续只传新提交输入的features，71,680B/input token。target阶段0/1各自汇出其负责taps，再由head整理；用ib0上persistent framed-binary socket到draft，不走公网登录链路。此socket不替代target现有IBext通信。CPU staging/网络时间计入cycle，不伪称GDR。
4. draft一次生成7候选；K=2/4/7只截取前K，不能因改K而重载草稿/target。使用相同完整span8 draft成本，先减少变量。
5. target验证输入[anchor,*proposal]，取K+1行target预测；最长greedy前缀a，发出proposal[:a]+correction。下一轮pending anchor=correction，不重复计算两次。
6. MLA KV只提交输入anchor+accepted proposals对应前缀，丢弃拒绝后缀。KDA recurrent matrix及短Conv不可像KV直接按seq_len裁剪：原pair API只有整个block的最终shadow，partial acceptance需从仍保留的active执行一次长度a+1的prefix replay后swap；全接受直接swap，EOS/达到输出上限跳过replay。实际native事务和位置255/256跨页、a=0/partial/all的覆盖还需本轮实测。
7. AttnRes bank/prefix/pending delta属于每个输入token的跨层前向中间量；不要把拒绝token的features追加到draft cache，不把草稿query KV当target context。只将被消费并提交的input rows缓存。捕获必须纯读，不能重复提交bank/delta。现checkpoint未声明attn_res feature stream；先保留997同一路径，若已有consumer pre-norm mixture可低成本同次捕获则做固定两套offline对照，不能为未知契约反复重载。[作者AttnRes说明](https://raw.githubusercontent.com/lightseekorg/TorchSpec/main/docs/kimi_k3_attn_res_capture.md)

第一版可以复用已存在trace buffer接口完成真实闭环，但其当前逐调用NFS落盘会计入cycle且可能严重拖慢。要取得有意义的速度，需要将固定五tap改成有界内存导出（仅必要host TU/CAPI），保持owner/stream生命周期；不扩大到全层trace。不对未执行的内存导出声称已完成。

## 测试与计时

已新增并执行`policy.py`/`verify_cpu.py`：greedy a=0..7、partial replay/pending anchor只消费一次、EOS4处、cap4处、稀疏q解析质量守恒；60000条双token随机序列与精确目标联合分布对照，最大绝对概率误差0.00312。CPU模拟仅验证算法与事务所有权，不替代KDA GPU状态和整模型质量。

同一次target加载：先必要forced commit/replay安全smoke；两个固定自然I512/O64提示，普通AR与spec K2/K4/K7交替测，包含warmup但不计warm性能；随后2–4个GSM8K任务以真实任务长度和足够输出上限查看最终答案，报告截断。45min hard/2640inner内预留清理；历史加载25–28min，预算不足先缩减提示数，不缩短cycle/隐藏截断，不擅自延长。这个小样本只是基本质量smoke，不称benchmark无掉点。

下一阶段在已证明有加速且稳定的情况下扩展至少64GSM8K+64MBPP/代码用例，统一chat template/采样/答案提取/运行器，对照普通target。报告配对正确率差及置信区间、失败类型、长度分布、截断率，不能用两三个样例证明无质量回退。质量阈值需结合基线、CI和任务，勿将固定token匹配作为替代。

主要指标：每用户useful committed decode tokens/用户decode wall（不含prefill，包含draft/capture/transport/verify/replay/commit/emit）；另报首末发出token速度、P50/P95/P99 ITL、burst大小/最长停顿、aggregate/card、TTFT及prefill input tok/s。cycle内分解draft、target verify、partial replay、features和网络、commit、emit；GPU事件与跨host wall分开，不求各卡时间之和。EOS不计useful output，拒绝后缀与重复replay不计产出。

用实测tau评估：TPS=tau/(Tdraft+Tverify+Treplay+Tfeatures+Tcoord+Temit)。997四段离线前缀若仅作条件例子tau=3.25，15tok/s要求cycle<216.7ms；现675 verify本身294ms，说明这组假定不够，但不证明其他任务/接受率的硬件上限。若tau=6则预算400ms，tau=8预算533ms；必须实际测出tau和完整cycle，不能承诺5–10x。仅增长K而接受长度不增长，会降低性能。
