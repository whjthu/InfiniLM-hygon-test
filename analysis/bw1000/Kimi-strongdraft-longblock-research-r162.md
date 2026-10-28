# Kimi 强草稿与长块首判定（2026-10-28）

结论：当前最短可执行路线是继续使用已驻留五层 DFlash2，在同一次必要目标运行中留出少量严格配对的数据，做 **同目标、同五个 tap、同 stream 的低成本继续训练首判定**。没有证据支持将现有 block8 改配置变成有效 block16；也没有找到可直接替换、已证明本机更强的长块 checkpoint。本文仅只读本地源码与第一方公开资料，无 SSH、Git、Slurm、新任务、包安装、权重下载、共享代码修改。以下时间均为工程估计，不能当性能/工时承诺。

## 可用 checkpoint 的实际边界

| 路线 | 第一方证据 | 对当前系统的判断 |
|---|---|---|
| LightSeek Kimi-K3 DFlash2 | [作者模型卡](https://huggingface.co/lightseekorg/kimi-k3-dflash2)：5层，H7168，tap19/37/66/78/90，block8，7 proposal；生产采样评估 | 已有 exact 模型家族，tokenizer/embedding契约已有本地证据；精确训练目标 revision、量化与历史 hidden stream 仍未知。无需再次下载/加载。没有作者已训练 block16 产物证据。 |
| Modal/Z-Lab Kimi-K3 DFlash | [作者模型卡](https://huggingface.co/modal-labs/Kimi-K3-DFlash)：可用 block16；generic mix，无agent/tool traces；推荐默认8 | **已是旧候选**，本地 c192 六tap/六层5.197GB权重已在库，且有线上负结论及后续离线无改善。不能包装成新强draft。近900 tok/s是C1、8×B300、实验verify后端；不外推本机。 |
| LightSeek Kimi-K3 Eagle3 MLA | [作者模型卡](https://huggingface.co/lightseekorg/kimi-k3-eagle3-mla)：专用于K3，训练/评估NVFP4目标、GB300 TP8，topk1/steps3/4总token，QA acc_len2.710 | **新增值得登记的无训练备选**，但这是短AR draft；不是官方EAGLE团队checkpoint，也不是已证明更强。没读到成功返回的config，具体tap/vocab映射/权重大小待定；不可承诺plug-in。应先用已知本机cost推导所需accept，再决定是否值得适配/下载。 |
| LightSeek Kimi-K3 DSpark | [作者模型卡](https://huggingface.co/lightseekorg/kimi-k3-dspark)：5层MLA、7proposal，tap7/31/47/63/87，强制attn_res | 不同五tap与stream，不能复用现有DFlash2特征。不是8/16长块答案。卡给出150K再生成样本、6891steps、40GB200合计1104 GPU小时，反证“随便小训即拿到公开质量”的承诺。 |
| EAGLE3通用/PEARL/SpecInfer/Medusa | 见下文 | 方法名不提供当前K3权重、AttnRes/KDA适配或免费verify成本消失。 |

本地证据：`analysis/bw1000/kimi-draft-contract/REPORT.zh-CN.md`及`research.json`记录当前 pin e77935fb；`analysis/bw1000/Kimi-speculative-current-feasibility-r8473.json`记录旧Modal线上W8/W16部分负证据（当时TP8PP4，不能将旧速度当当前速度）；`evidence/bw1000/review93/Kimi-block16-remote-metadata-r9112a.json`记录旧权重存在。与当前用户质量政策冲突的历史exact-token资格门槛不沿用。

## 不能无训练获得“有效长草稿”

[DFlash原论文§5.4.4](https://arxiv.org/html/2602.06036v1#S5.SS4.SSS4)直接测试train8→test16：平均接受长度没有改善；train16→test8较可行。它是普通DFlash上的经验，不是DFlash2/K3定理，但足够否定免费扩块保证。现有代码还写死block8/7proposal，Native每row≤8、totalT≤64；因此即使离线可输出16，也需要验证长度、KDA状态保留/拒绝回放、边界bonus、EOS、row账本一起扩展。两段W8验证最多是分段验证，不能声称“一次长块verify”。

字面“8/16个新草稿token”通常意味着W9/W17；本项目W8/W16分别含anchor，所以只有K7/K15。报告务必分清。

## 最短低成本训练实验（建议，未执行）

1. **先获益上界，半天内可完成CPU准备。** 使用已有3304cycle的每题accepted、W8 verify wall及C1/C8/C16数据，计算保持实测verify成本时所需平均有用token数。对拟议新draft，只接受大于成本门槛的预测；短Eagle3公开acc_len不能跨目标量化/采样直接代入。当前慢题“去掉其他开销仍13.07tok/s”是当前接受轨迹的floor，不是所有更强draft的全局上限。
2. **同一次必要目标驻留，采64–128条校准轨迹，首判定用现有33卡。** 中文QA、代码、数学、写作、长上下文分层；冻结一半作held-out且不用于选长度。记录target revision/量化、tokenizer和chat template hash、tap层/前后norm/AttnRes consumer、input IDs/assistant loss mask、EOS。只导出必要五tap；raw activation留实验端。本次没有采集。
3. **先做block8继续训练对照，再做block16。** 优先warm-start当前checkpoint，全backbone+unary+selector，block16同时监督后半块并保持前缀权重；不是只改config或只训selector。保留block8 control、相同数据/steps。初始用1–5M训练token、100–500 optimizer steps作为能否学习与wall收益的首轮筛选，而非最终质量配方。数据不足或held-out接受长度不改善即停；不许用teacher-forced逐位置准确率冒充自条件路径接受率。
4. **资源/时间真实边界。** 3.69B BF16参数约7.39GB；Adam训练仅参数、梯度、FP32一二阶状态就约44GB，FP32 master另约15GB，尚无激活/embedding/head/临时buffer（推算）。故1张未知显存DCU不能预设可训；需gradient checkpointing、冻结共享embedding/head，必要时2–4训练DCU分片。可先捕获后释放32 target卡离线训，再回到必要target驻留验证，避免将40卡在线recipe机械搬来。到首学习曲线预计1–2工程日，**首真实wall结论预计2–5工程日，前提是已有可运行的Hygon训练栈**；当前没有该栈、step吞吐、数据生成吞吐证据，因此不能承诺几小时或一晚完成。
5. **先block8改善通过后再升级Native长块。** C1/C8/C16/C32都配对报告每用户decode、TTFT、ITL P95/P99、总有效吞吐/33卡、失败/截断数。totalT64当前可容纳C8×W8，但容不下C16×W8，C32更需分波；block16在同一cap下只容4row。扩块若让分波更多，可能提高C1却伤高并发。必须同一轮cost曲线验证，而不是公开倍率推断。

训练可参考作者[TorchSpec配置](https://github.com/lightseekorg/TorchSpec/blob/main/configs/vllm_kimi_k3_dflash2_mla_swa_gb300.yaml)和[selector训练源码](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/models/dflash2.py)。公开recipe是6层6tap，与当前5层5tap不同，不应当作现checkpoint生成记录。[Speculators教程](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/tutorials/train.md)提供online/offline与on-policy数据流程；其[DFlash2说明](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md)明确自身目标是experimental，不能声称复现原始训练。

## 质量与lossless的区别

早期验内容合理、数学答案/代码测试/检索正确、无重复崩坏、EOS/截断完整；后期用冻结基准验证无明显掉点。建议首轮≥200条任务混合，最终MATH500、GSM8K、HumanEval/MBPP、中文/多语QA、长上下文检索、工具参数/JSON校验；同时给AR基线分数、差值和bootstrap置信区间。先预设如绝对下降≤1个百分点的目标，但样本不足以分辨时应写“证据不足”，不能宣称无损。无需exact tokens通过。

分布lossless是算法性质：[原始speculative sampling](https://arxiv.org/abs/2211.17192)在每步用真实proposal条件分布q，以min(1,p/q)接收，拒绝时从归一化max(p−q,0)校正分布采样；temperature/top-p等变换都需在实际p/q中一致。DFlash2 selector改了proposal law，不能拿原始unary softmax充当q。greedy proposal可以视作delta q，但要对应正确校正逻辑。正确实现下可维持目标分布，却不要求与另一次AR随机运行token一致。

若接受target top-k内、概率超过阈值、或“语义差不多”的草稿，这是近似接受；可在本用户允许下实验，但必须明确分布已改变，质量用基准负责，不能同时称lossless。即使精确验证，量化/算子变化也只能保证对应实现的target分布。

## 为何公开大倍率不能直接套用；其他方法是否合适

不能一概说论文只有B1。[DFlash论文§5.3](https://arxiv.org/html/2602.06036v1#S5.SS3)确有C1–32的单B200+FA4+Spec-v2结果，收益随并发下降；其§5.4亦显示大tree在高并发可能慢于AR。这说明权重带宽空余允许多token并行时投机好用；高batch进入计算/激活/路由/通信瓶颈，verify成本与草稿/拒绝成本占比会升。当前32卡TP16PP2、KDA、MXFP4与verify CAPI瓶颈完全不同。

- [EAGLE3官方仓库](https://github.com/SafeAILab/EAGLE)：融合多层特征、自回归draft；树能提高命中，但增加target节点，KDA树状态分叉也尚无本机接口。仅建议登记LightSeek短链checkpoint备选，不做大树移植。
- [PEARL作者代码](https://github.com/smart-lty/ParallelSpeculativeDecoding)：通过pre/post-verify和draft/target并行减少等待，不能免费增加当前block8已训练跨度；本机主耗时是verify时，隐藏约20ms draft的上界有限。还需解决投机上下文和KDA状态生命周期，优先级低于真实verify成本改善。
- [SpecInfer作者实现](https://github.com/flexflow/flexflow-serve/tree/inference)：多个小模型+树验证，公开支持架构列表无K3；需要树attention、KDA分支状态、可用小draft，移植成本高，非短期答案。
- [Medusa作者代码](https://github.com/FasterDecoding/Medusa)：额外预测heads+tree，典型接受可改变目标分布；可训练但不存在已识别K3官方head，且无法消除当前verify地板。若以后做近似accept，应作为独立质量实验，不能当保分布优化混用。

本轮真实阻塞：当前没有同目标同stream block16新权重；新Eagle3 metadata取回未成功（网页仅模型卡，shell网络受限）；Hygon训练栈/训练显存和吞吐未资格；Native W16/T>64与多请求KDA回放未资格。以上都不能靠论文标题或“换个草稿”解决。本轮没有触碰运行作业或敏感参数。
