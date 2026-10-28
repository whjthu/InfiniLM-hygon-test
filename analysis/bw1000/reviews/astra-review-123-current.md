第123轮复核：继续性能优先，目标尚未完成，未证明达到硬件上限。此次是提前复核，沿用root截至触发时的GLOBAL GPU两轮计数；本报告不把CPU合同、组件结果或归因测量凑成第三轮。

- **采纳** GLM781的受测收益：task 14.262→15.827 tok/s（+10.98%），每卡0.891→0.989；decode HOST CAPI +17.34%。用户p50仅约0.861，所有用户仍低于15。32题四臂各31正确、32EOS，不等于广泛质量不降。
- **第一优先** 消费已完成Kimi449：target/draft/keeper均正常、audit0、postempty。真实MoE子段已取得运行资格，接着离线分解并只选一个有整模潜力的候选，禁止再为提取归因加载。stage1 whole含incoming，不能与stage0 whole相加；L1/L47稀疏样本不直接外推全层。
- **采纳** Deep158一次n05/8卡加载，CPU500已过。B32→8→16→4、同64题、prefillmicro4可筛选用户/总/每卡折衷。B4 cap2 warmup未温热B32/B16，单遍顺序不能证明最优或增益；有清楚候选后最多复测候选与基线一对。F100与生产ac937分开；1165固定synthetic阴性不清除整个F100模型链。
- **采纳CPU、暂缓专门重载** loader157：Linux共享global-source、8owner、真实TP16几何成对复测；下一次必要加载记录CPUtime、majorfault、IO，并拆分gather/state_dict/H2D。Windows1.80x仅组件结果；23亿blocks不是23亿atomic。r9212d另行归因，不同时合并。
- **暂缓专门重载** GLM158。源码开关同时改变chunk128与PREFIX/ANCHOR后的decode优先，应称“策略包”。虽仅5题被拆分，其他ready行也受调度影响；PREFIX4→7、主要阻塞仍长，收益未知。只搭载下一次必要GLM加载，保持两臂deferON。旧chunk32停止、P0 GPU重载暂缓。
- **停止** Kimi直接两micro overlap、W16/容量扩展、GPU F64、无判别力的逐位追踪、删除atomic微调、以未分配物理RAM规划全checkpoint常驻。

Kimi下一步按真实layer/rank descriptor计算 `F=2Σm_e(K13N13+K2N2)`，FMA计2；fusedW13与W2若实证为H→2I及I→H，才简化为 `6HIΣm_e`。packed FP4、scale、activation、临时量及collective字节分别列，逻辑bytes不当实测HBM流量。HOST/stream区段所得仅inclusive effective TF/s或TB/s；333.752 TF BF16与1.23711 TB/s是achieved reference，不是物理峰值。只有专家重用/不均衡会影响候选选择时，才在原drain后加真实histogram；先区分TP复制路由与EP分配，不能把expert skew直接当rank负载不均。

完整实验卡、首个判定成本、条件上限与证据见同名JSON。单靠MoE零成本的用户p50约11.875只是q=.89等假设下的条件式，不能证明系统无法超过15。未新增质量门，也不要求输出token identity。


同123轮增补（9451）：真实4个T64窗口×32owners已qualified，最大正observer扰动0.654%。W13第一、W2pair第二：L1/L47 W13为3.3607/3.0095ms，W2为2.0821/2.0730ms；用2.8186/1.4093 GF每rank归一，分别约0.8387/0.9366 TF/s与0.6769/0.6798 TF/s，均是含解包/Situ/依赖的effective率。可见15区间和为6.32675/5.94298ms，W13+W2占86.03%/85.52%；仍遗漏FFN.begin→router.begin，不能称完整全和，更不能直接外推全部层。ON输出6287 token而两OFF6383；39.167/39.434/39.204 tok/s不证明算法提升。

下一步明确先做一个W13同expert两token小tile复用FP4解包与scale的组件候选，scalarFP32/Situ/布局保持，单token和尾部fallback；先确认源码没有已有等价复用。真实U/J未知，组件先覆盖有限m_e=1/2/4，synthetic只能称组件资格。449已退出，不能追溯读取counts，也不为histogram专门重载；下一次必要模型在原drain后读真实counts/U/J，再判断tile复用比例是否足以进入唯一整模对照。若单token主导或组件无净收益，停止此tile，不自动改W2。详细候选与解释见JSON增补。
