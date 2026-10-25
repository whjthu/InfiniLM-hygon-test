# Deep1056：实际Model EOS行压缩结果

TP8/PP1单节点8卡，GSM8K固定32题/custom3shot，自然EOS，max512。唯一加载后测试B8/B16/B32各fixed/compact两arm。

|B|fixed decode tok/s|compact decode tok/s|变化|compact tok/s/card|
|---:|---:|---:|---:|---:|
|8|12.1238|10.7913|-10.99%|1.34892|
|16|19.9594|19.9191|-0.20%|2.48989|
|32|27.1382|31.7278|+16.91%|3.96598|

B32为本轮明确正向范围；B8变慢、B16持平。不能称普遍改善、单用户速度或HTTP服务吞吐。prefill时间和输入吞吐单列在canonical JSON；B8先执行fixed的首prefill含cold/初始化影响，不能将该prefill差异当压缩收益。

每个arm都32/32正确、32EOS、0cap，finite/readonly/canary与packedprefill命中均通过。192是6arm的评估次数，独立题目只有32；不证明无质量掉点，B32也只有单cohort/arm性能观察。

910374.1056 COMPLETED0/33:27，加载441.39s；原destroy/join报告通过，post1057KFD空。未保留每owner独立normal marker，完整证据是原engine destroy/join合同、报告及外部post清场。CPU auditor return0且绑定原报告SHA f69c1a26…；旧失败/qualification保留。
