# Astra review 129 — 工程提前复核

新增可证明整模型/真实服务速度收益 **0**；128 后性能轮次 **0**。本次由反复可避免的审计/加载器错误触发。原三模型性能目标保持 active，用户明确优先的正常智能体验收合理且不降标准：32K 上下文、普通4K/思考8K输出预算、8K/16K实际输入、至少8次真实工具、至少1K实际工具参数及C2隔离。

1. **收益与证据**：Kimi542六HOST构建、GLM543一HOST registry修复、Deep1200的35边界/200chunk metadata均是工程进展。GLM792的32个owner全部在create遇到TP8PP2 identity断言，未测得OOM、weights fit或forward。没有新的可比性能配对。
2. **时间**：当前消耗集中于源码准备、审计和加载器闭合。可见Kimi542构建97.506秒、GLM543 13.639秒；Deep1199/1200分别15.306/30.403秒失败且编译0 HOST。端到端准备、排队和实际GPU测量分项不足，记UNKNOWN。资源曾可用而任务未就绪；root独立推进r188正确，不应等本报告。
3. **profile适用性**：旧GLM787 TP8PP2短T1和Deep1195短C1不能预测当前32K工具任务/GLM PP4。已证实的是registry拓扑拒绝和Deep嵌套quoted include混旧guard。Kimi8K/16K分别960/1984次追加verify/fullcommit是源级成本风险，GPU计算、通信、空泡、CPU调度各占比仍UNKNOWN。
4. **保留与停止**：保留GLM最小caller修复、Deep私有依赖镜像和Kimi542实际产物；停止重复CPU证明、短smoke reload和新的审计框架。128性能候选保留到接口验收后；Deep smallT继续STOP。

**GLM：采纳r184b→r186/r188，先消费当前句柄。** 内部identity新增pp，public类布局和数学未变；原PP4边界0/18/38/58/78，PP2专用优化在PP4明确关闭。543已闭合1 HOST/24复用对象/ABI/动态链接，无需重做。792没越过create，新的32owner constructor/census/trim/通信/退出判定有必要，不属重复容量探针。512MiB余量只代表constructor探针，不能代表weights/finish/long-prefill峰值。通过后直接完成一次正常服务加载与全部冻结客户端。

发现一项**下一次启动的效率/容量风险**：r186模块顶层执行pv/ev，spawn的8个worker会重复整套历史preengine，且先初始化GPU0再由worker切换rank。静态上存在每节点父进程+8子进程重复审计和多份GPU0 context，未证实本轮有故障。不要打断在途r188；结束后将纯元数据祖先审计放父控制一次，子进程先选择自己的device再做该进程必需的DSO/symbol绑定。不要全部删除跨进程绑定，也不要为这次整理新增GPU门禁。

**Deep：采纳r180d，一次完成HOST闭合。** 镜像全部真实项目依赖、保留普通source/header字节和目录结构、私有symlink别名、归一化依赖相等并禁止逃回旧树，确实对准嵌套quoted include根因。保留现有fresh-M/MD、全部guard/RTTI/vtable消费者替换、最终DSO无旧message和ldd；不要再扩展测试。当前依赖闭合不能冒充旧object编译时header已知。正式32K/rows1/1024chunk，per-forward4096不变；64K只非阻塞扩展。200chunk metadata不是HCA/CSA/index/compressor数学证明。CPU成功即绑定真实服务并保持4K/8K预算，不再跑短smoke加载。

**Kimi：复用542，直接准备真实容量与一次常驻服务加载。** logical1/physical2/pool3/128pages/260blocks对应新shape；0 HIP改动不等于KDA pool3、conv历史、MLA长位置RoPE已验证。真实prompt先512再width8全commit，没有按预测过滤输入。复用已有long_prefill_observations，在同一次正式load先观察实际8K/16K墙钟及peak，再继续冻结验收。27请求保守35136次调用，8500秒即使忽略load/decode也仅允许平均约0.242秒/次，不能盲等最终超时。若真实成本不可承受，在安全fullcommit边界取消并保留证据，才启动已有scoped append512替补；64倍RPC减少绝不是64倍速度声明。

逐模型各一张完整执行卡（owner、资源、假设、最小改动、数值/生命周期检查、样本、预计首判时间、退出/替补和时间记录）见同名JSON。排序GLM→Deep→Kimi按准备度与首判时间；三者速度潜力均UNKNOWN。GLM实际容量预计2–12分钟，Deep一次CPU闭合2–15分钟后历史load约7.5分钟；Kimi应在30分钟内形成绑定542的可运行adapter/controller，历史load20–27分钟而长prefill时间未知。这些为执行估计，不是已测时长。

根建议：采纳三张卡；延后append512、64K和128性能GPU；拒绝重复同源码CPU/同配置探针或缩标准。GLM重复preengine问题留给当前句柄结束后最小整理。接口任务是用户明确优先，也是未来有效服务性能基线；它的工程收益与性能收益必须分账。报告仅提供建议，不替根发布、启动或登记实施。

根增量修正：r186/r188已发布于d82769a2ce6544363e9c493642bdf3299177bd54，目前上传准备，尚未实际提交；上述“在途/消费句柄”指随后同一次实际构造。先按已发布固定包做这一次，不前置新修改。rank0若因额外初始化context触floor，应归为harness开销，不能直接判32K Native不fit；通过也仅证明包含该harness开销的constructor fit。

正式service前最小启动修复可在Python层完成：preengine.validate接受device并用rank；把worker Native/DSO初始化移到worker开头set_device(rank)之后，禁止spawn重入顶层GPU初始化。父端全量preengine在现有controller的一次短命子进程完成并退出，再spawn；若生成communicator IDs也要求父端Native dlopen，则同短命初始化过程生成并返回IDs。每进程symbol-owner绑定仍保留，静态祖先hash/旧fixture仅父端一次。无需重编Native、追加同配置constructor探针或新框架。
