# r171 封存交付：CPU优先，GPU_GO=false

用户已明确智能体接口验收优先；本包封存供后续续跑，不提交任何作业。实际父CPU533 receipt为`3729c329987ddf2c270f43e51b6ebb95de143847da8687daf66ef0712cf8332e`，r170 Native为`06db1d3ad90751e552b9a3c4f62638de8f78465aafeef5b3b528c6c2f8373b20`。

root审核发布后，可仅部署到`/refactor/incoming/Kimi-persistent-subset-r171`，在既有masked CPU keeper执行`bash CPU_inside_r171.sh`。它只编译一个HOST MoE+probe TU并链接新Native，复用CPU533原cache.o和其余33对象（包括cache.o），不重HIP、不dlopen、不创建Runtime。新输出`/refactor/candidates/Kimi-persistent-subset-r171`必须不存在。当前active kv_cache.cpp的无comm validator源码SHA必须吻合冻结证明，否则CPU止步。

`probe-prefix-r171.inc`是原r170 MoE源码，只将packed原body逐字逆映射提取为同TU生产共享helper。真实Experts构造原comm检查、r170实际lease赋值/消费路径均保持。probe CAPI就在该MoE TU，实际生产与probe共享唯一registry；原r170 Native不同时加载，不通过另一wrapper DSO实例化第二套registry。实际r170 cache.o逐SHA复用，runner分别对probe入口与cache factory做dladdr+Native SHA绑定。

待未来另行GPU授权后，受限单卡任务才允许直接创建92组exactbanks（368storage/494592leaf）、完整93层cache并全字节零读回、92字段共享workspace。owner A/B/C均由同一owner线程创建使用；其他线程只测试真实Lease的preGPU拒绝。实际Lease正常及host异常unwind通过同stream memset/D2D/event/pattern验证；弱owner/Tensor过期和新generation亦检查。不造comm，不加载权重、不forward、不注入GPU故障。

逻辑43.112GiB只是持久子集；按实际cudaMemGetInfo每组记录端点并守4GiB floor，bank前另预留3GiB、cache前2GiB。没有618MiB单组乘92的物理推断。未测全参数/router/head/AttnRes、正常算子、load/finish或CCL高水位；Native pool reserved未知，Runtime销毁残差归属未知。r170每I96新增sync的速度风险不变，fullModel维持UNKNOWN_STOP。

包内`GPU-release-template-r171.json`默认false。**不得直接运行controller**；接口验收后须由root独立审核实际CPU receipt，再另存一份外部release JSON明确授权/发布commit/CPU receipt SHA。controller仅接受该外部release；固定CPUkeeper407控制parent374/n01、fresh/post KFD、单GPU独立新进程，Slurm4min/inner180s。host控制器在`/public/...`原host运行，GPUinside经既有outer容器运行，三VISIBLE masks=0、完整CPU receipt LD依赖，不importTorch、不检查IB/CCL资格。当前没有GPU放行。
