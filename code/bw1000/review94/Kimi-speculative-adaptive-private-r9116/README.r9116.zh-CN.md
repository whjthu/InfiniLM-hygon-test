# r9116 前端交付：状态重放、回退、驻留队列

独占源已冻结，r9046 原实测源不改。native 唯一 launcher owner 为 r9114。

CPU 实际四节点线程+真实 binary socket 的24case回归通过，数学为 mock，不构成 GPU 性能或质量证据。每节点418次 state-only replay、22次旧整模型 replay、342次 pair-AR continuation；双草稿、六tap、两轮回退、无AR基线校准、驻留追加、stop、512输入+512输出末轮缩宽均验证。

集成时在加载权重前将本目录放在 sys.path 第一项，再 import target_protocol；target_binding 继续由 r9114 私有原生绑定提供。主入口 run(g) 沿 r9046 的 globals 合同，原 experiment_started_r9046 记录加载前时钟。manifest 应冻结本目录实际运行的6个源/样本文件及 r9114 DSO。勿在驻留时改源码、DSO或SDK。

初始16case顺序为两GSM8K各 AR/oldspec/state/adaptive，再两个自然prompt各四策略。K7，GSM cap256，自然 cap64，首先取得真正 state-only cycle。oldspec 仍用原完整模型前缀重放，state/adaptive 使用 r9114 owner replay 并复用原verify的feature前缀。两种均是原fastW8/full-logits flags3/Math0，没有历史AR输出逐位门槛。

adaptive 完成事务后计算 emitter 已输出 useful tokens / 从首token起实际wall，至少2投机cycle后若低于本题AR*1.05则永久回退至 pair-forward W1+commit0。不会reset已有state、不再查询草稿、不重复消费anchor。没有基线时另记32token AR校准case。pair-AR continuation 与原ordinaryAPI基线分别记录。

设置 KIMI_CASE_QUEUE_R9116 为唯一安全basename，leader从 /refactor/results/<basename>/q*.json 按序读取不可变JSON并广播。具体schema见 protocol-contract-r9116.json。qfile应先写临时文件再原子rename成q*.json；唯一id，数据不是任意Python。stop显式结束；否则 KIMI_PROTOCOL_BUDGET_R9116 默认5000秒（从加载前计时），预留150秒cleanup。draft alarm/accept/idle recv都须覆盖5200预算，原2640/180秒不可沿用。

native 总context1024：prompt<=512、cap<=512且prompt+cap<=1024；不截断题干，超界记录capacity_skipped。每轮effectiveK<=remainingoutput-1和1023-past；最后1token直接W1，避免K0草稿或越界verify。现 Modal 仅可选K<=7真实W8，不声称W16或chunkedK15；六tap显式begin_r9112，相关env见合同。

pending-queue-random32 是root原seed20261026官方固定随机32题的64条 AR/state 待追加命令，尚未提交，cap256。root可在 staging 前将配对预算一起设512并记录新command SHA；必须统计EOS、截断、coverage，不能把短cap正确率当benchmark无掉点。90分钟剩余预算不足时报告完整paired样本数与pending，勿伪称32全完成。

目前emit为native callback，token-ready时间在feature传输/commit/replay前；尚无外部HTTP stream hook。服务adapter不能将case结束响应假称逐token流式。decode useful numerator排首prefill token/EOS/拒绝草稿/replay输入，wall含全部真实cycle；prefill/加载另计。新增schema/API见合同，真实指标须等新GPU步骤。
