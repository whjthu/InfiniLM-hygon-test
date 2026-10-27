# Astra review 120：三真实模型 P0 优先，先修两处确定接口错误

结论：ADOPT 三真实模型 HTTP/SSE/tools P0 优先；小 Qwen 不作前置，Kimi C8性能probe暂缓让出32卡。P0是当前用户优先级调整，不是原有吞吐、质量、单用户>15目标已完成。三模型保持原qualified Native数学及依赖，本轮不夹带量化、投机、显存池或调度优化。

本审核只读本地证据和源码，写本报告，未Git、SSH、编译、GPU或改源。两项确定错误已立即发root，独立预检不等审核：Kimi输出上限缺少HTTP前置校验；共用bridge超时原因被覆盖且非流式可误报正常stop。以下判断基于所读冻结版本，owner后续修复须登记新SHA和验证结果，不能把建议视为已修复。

## 119 落实及新增波次登记

| 唯一句柄 | 新结果 | 资格边界/决定 |
|---|---|---|
| GLM 910376.767 | COMPLETED，keeper459正常，post768空；全16owner、128自然selector，G119A闭合 | 一轮真实全模型诊断，非新加速。G119B STOP，patch=0；不为填候选再加profile/load |
| TP32 910375.767 | constructor normal32，GPU46秒/keeper56秒 | GPU构造/容量资格；不是full93权重加载、forward或服务吞吐。full93继续UNKNOWN_STOP |
| Pool 910379.487 | normal16、postempty；ON原生回收814MiB，driver回升784MiB | 一轮显存回收与成本因果诊断，历史重复输入，非478实际激活、非整模型提升。默认OFF，entryshape142仅plan UNREADY |

三个GPU完成事件可登记本轮消费，必须分别标注诊断与资格，不能写成“三轮整模型性能提高”。新增qualified整模型加速结果仍为0。

GLM controls-before/profile/controls-after task分别13.170224/13.243602/13.197909 tok/s，decode22.716397/22.710468/22.606088；均31/32正确、32EOS、0cap/unfinished、2215visible tokens。漂移task0.2100%/decode0.4868%，profile效应0.4516%/0.2172%，都在原3%内。stage0 T16本地约243.49ms与stage1 incoming244.47ms对应；这是必要生产依赖的持续时间证据，不是可删通信。最大coarse单层仅同owner CAPI3.916%/emitter2.367%，代表层detail不能乘层数。没有>=约10%具体可改critical项，停止G119B是正确落实；不能据此断言模型已达硬件极限、专家均衡或CCL零成本。

Pool ON计数回收853540864bytes=814MiB，driver after增加822083584bytes=784MiB；不同记账口径不得混称。sampled min始终65506639872bytes，未改善first/live lowwater，且采样不是精确峰值。暖态OFF约6.3ms，ON约18.8/28.8ms，紧跟ON的OFF约19.6ms，后续OFF回到约6.2ms，存在重分配carryover。不能用简单ABBA平均声称稳定速度收益；不能把回收量乘93层，也不能默认每层trim。2K容量仍PARK，只有明确原完成边界的entrytrim最小源码成熟后才再讨论组件验证，本轮不给它抢P0资源。

Deep最新1138实际GPU COMPLETED/9m23s，8owner正常close、postempty、all_state_cases_passed和四个diagnostic controls完成；numerical_quality仍UNRESOLVED、model_quality UNQUALIFIED。keeper469 FAILED/9m37s是audit assertion问题，不能覆盖GPU真实完成，也不能把state通过改写为数值/模型通过。修keeper字段/判据需保留原失败证据，零GPU；未查清前不登记整体qualified。原prefix分支无MTP head与spec收益。P0使用已qualified ac937普通AR loader/forward，与未晋级prefix事务分支分开，不因prefix数值未解就拒绝独立普通AR真实baseline。

## 确定错误：只做这两组最小修复

**R120-1，Kimi输入资格缺口，P1。** 共享bridge d3bc4904的validate_data/add_chat_request只检查prompt+max_tokens<=1024；Kimi target_protocol.run及native_ar_decode_r144b.generate都assert 1<=cap<=512。短prompt配max_tokens=513可经HTTP接纳、广播到四node，随后在请求try/finally之前断言，导致resident服务退出。当前默认128正常P0不触发，但这是可复现的支持范围错误。最小修复为Kimi模型专属max_new_tokens=512前置validator（或bridge可选上限参数），同时用于pre-SSE检查与enqueue，worker保留防御检查。只做CPU纯输入513拒绝/512符合总context时通过、prompt512+output512边界检查；不能仅删除原生assert或把请求静默截断。目标15分钟内闭合，不重编Native。

**R120-2，共用超时语义，P1。** stream_request在queue.Empty超时先设置TIMEOUT并yield finished timeout，但finally在owner还没finish时又用CANCELED覆盖abort_reason。原InferenceServer._chat看到timeout便break，然后读取尚为None的req.finish_reason，可能回HTTP200空正文、finish_reason='stop'。持续progress时request_timeout也不会检查，因为绝对超时判断只在queue.Empty分支。三份bridge当前SHA一致，问题共享。最小修复：每次循环检查绝对request deadline；保留首次TIMEOUT/ERROR原因，disconnect只在未有原因时标CANCELED；非流式超时明确返回timeout错误或明确timeout finish，不把None映射为成功stop。不要为修HTTP结果提前req.finish，Native reset/drain仍由原owner完成。短CPU模拟覆盖连续progress越总时限、无progress超时、取消后reset完成与后续请求恢复；这只是bridge资格，不是Native通过。正常P0不必等广泛压力测试，异常恢复项在修复前只能FAIL/UNQUALIFIED，不能勾通过。

另一个相关但非当前P0阻断的边界：bridge finally会在owner仍处理/取消时移除_requests，因此max_pending不是严格的原生队列容量；本轮只验串行P0，不宣称过载/并发服务保证，不为此扩展压测或改调度。

## 正常路径、线程与加载静态审查

共用existing_server_app构造InferenceServer只保存配置，随后显式server.engine=bridge、自建FastAPI lifespan并调用原_register_routes。没有调用原_create_app，原lifespan中的AsyncLLMEngine构造不会走到；当前静态源码未发现第二模型创建。标准server import会加载Python模块/DSO，仍须实际preweight maps确认，CPU fake-worker或词表检查不能替代。

GLM的HTTP线程仅提交/消费队列，run仍在原worker(rank) caller线程调用CAPI；正常请求finally先原torch synchronize、model_reset、world barrier，再req.finish。排队时已取消且从未进入Native的请求可直接结束，不需要伪造一次reset。异常reset/barrier失败必须整组停并保留证据，不能出成功finish。Kimi Node0仅HTTP协调，四node原Native命令集合同步；请求finally native.reset+control.collect全部完成后finish，失败req.fail亦在reset尝试后，Native不迁到asyncio.to_thread。Deep采用ac937原loader/forward AST，原finally destroy保留；HTTP单独线程，serve在原caller执行CAPI，reset之后finish，reset失败标_fatal并停。以上为静态顺序通过，实际正常关闭及异常恢复尚待真实P0。

processor兼容性采用所需接口即可：当前HTTP routes直接依赖bridge.add_chat_request、stream_request、is_healthy、request.prompt和usage方法；它们已存在，不要求bridge伪装原AsyncLLMEngine类或构造原processor。GLM直接官方tokenizer template；Kimi frozen official K3 segment-aware XTML；Deep官方DSML encoder与tokenizer。prepare_agent_request把顶层tools放入chat_template_kwargs，三路径均消费该kwargs；不能在CPU tokenizer成功后就声称Native工具生成成功。未发现必须新增processor实例的静态阻断。

GPU preweight范围：Kimi在bind/verify_maps之后导入真实server/codec、创建注入app，再verify_maps并四node barrier，之后才create/load；这是目前最完整的防换栈路径。Deep先CDLL精确ac937并dladdr核7个实际CAPI，再真实server import和required DSO maps对照，才exec原loader；CPU1142 ldd环境失败应沿143b最小环境修复，零GPU，不换provider。GLM preweight核真实server来自私有overlay，实际worker加载前还有libinfiniops及Common maps检查；这些checks部分在constructor之后，不能写成“所有provider已在constructor前核验”，但可在权重读取前排错。CPU470通过只证明静态/源码准备，不是GPU import通过。检查失败即保留具体module/provider差异，禁止靠换Core、隐式pip/全栈重建来消除错误。

所有Native数学保持5f91ad/current Core实际对象；7ceb仅公共协议层及必要Python接口overlay。Runtime导入成功、model ready文件、HTTP health200都不是工具P0通过。

## 最短真实baseline与P0验收

已分配56卡/7node：Kimi380四node32卡、GLM376两node16卡、Deep374一node8卡；r9376 freshKFD空只是当时资格，不能推断永久空闲。本审核不启动。小Qwen139 DEFER，Kimi136b性能probe即使sourceREADY也DEFER，三模型独立预检/加载继续，不相互设前置。

每模型只一张P0卡，原owner负责：

| 模型卡 | 最短首判与复用 | 成本/退出门槛 |
|---|---|---|
| K120-P0 | 修R120-1/2，1K Native305726/cap8、TP16PP2、32卡；无draft。先一条完整真实ordinary baseline，再原HTTP四tool闭环、SSE同闭环和既有auto/required/negative/recovery | bridge源码修复目标15min；queue/load UNKNOWN；首baseline ready后目标<=3min（预算而非预测）。baseline异常先保留一次原生trace，停止后续工具链，不改数学；默认128输出，prompt<=512、context<=1024，不截历史 |
| G120-P0 | 原ce830c Native/640ctx/32rows、TP8PP2、16卡；CPU470后GPUpreweight真实导入，原caller串行AR，使用原client | 准备复用，不重编；queue/load UNKNOWN；ready后先ordinary一条，再原计划。实际tool历史超640则明确HTTP400并记当前容量不覆盖该workflow，不截题或挪首token |
| D120-P0 | 最小143b ldd环境修复，ac937、1024ctx/128rows、TP8PP1、8卡；普通AR，不用新prefix事务/MTP | CPU环境目标15min首判，queue/load UNKNOWN；真实preweight先核DSO，ready后ordinary一条再原client。若baseline本身不合理/非有限/状态错即停；不能拿1138 state资格替代 |

一次加载跑完已有P0计划，不新增性能网格。每模型至少需要：真实请求/响应原文和model/tokenizer/Native来源；普通问答基本合理、自然结束或如实length；实际生成完整schema-valid tool_call，真实受限工具执行产生结果，再按tool_call_id回填；四步read→calculate→write→read及最终随机nonce/sum闭环；HTTP与SSE各自证据、usage及finish；invalid schema/history与截断不得执行半个工具；最终恢复请求成功、reset和正常shutdown/清理。named四步证明受约束流程，auto/required单独记；conditional multicall未出现不能当已验证并行工具。不能注入预写答案、fixture输出或将83协议用例作为真实生成结果。既有client可直接复用，不另造框架。

Kimi真实官方词表计数211/301/392/497/438全<=512只证明这份模拟历史有容量余量；真实生成工具arguments/reasoning可能更长，实际每轮必须按原请求完整编码，超限拒绝并保留失败，不能用模拟计数替代最终链路。GLM/Deep亦遵守完整context预算。小样本普通问答和四工具通过是P0基本合理性，不是GSM/MMLU质量不劣证明。输出token可不同，无GPU F64、无逐位身份硬要求。

三份bridge目前都buffered：emit_token只存ID和progress，直到Native正常reset后done才一次decode并yield完整文本。SSE有合法role/data/finish/[DONE]/usage并不代表增量token流。只能标记“真实模型生成+缓冲式SSE协议资格”，不能报告用户可见TTFT/ITL或逐用户decode TPS；首个role空包不是首内容token。内部token_monotonic可作Native内部诊断，不能冒充客户端到达时间。client也一次response.read获取全文，未记录逐chunk到达；生产增量服务能力仍DEFER，不在本轮靠时间戳包装成达标。端到端请求wall可如实报告，performance_gain=UNKNOWN。

## ADOPT / STOP / DEFER

ADOPT：三真实模型P0优先；两处确定接口修复；Deep keeper469实际字段审计最小修复；逐模型真实ordinary baseline后一次加载消费现成P0 client；保持caller/reset/DSO绑定。

STOP：G119B无候选仍新增profile/load；Pool全93层默认trim及814MiB倍乘；TP32 full93推论；CPUfake/词表/83协议case→Native或P0通过；缓冲SSE→增量TTFT/ITL/TPS；超时空内容伪装正常stop；修接口时换Core或加载第二模型。

DEFER：小Qwen139；Kimi C8profile136b与动态W/量化；entryshape142未成熟实验；Deep新prefix数值晋级/MTP性能；增量流式和并发吞吐SLO。原性能目标保留，P0首轮结果出来后再恢复最有依据的性能卡。硬件exactSKU物理峰值仍UNKNOWN，717TOPS实测或参考带宽不能替代。

审核期间root已明确采纳两项错误修复：GLM公共bridge owner统一绝对deadline/首次TIMEOUT原因/非流式HTTP504并同步三模型，Kimi同时前置cap<=512；当前三服务GPU均尚未启动，修复后再启动。此为已收到的落实决定，修复源码及实际验证尚待owner证据。除此之外，所读冻结源码未发现其他确定启动阻断；实际GPU preweight仍需执行。

root须登记其余实际采纳/拒绝/暂缓与owner，报告不替代派发。源码修复后刷新各包manifest及已实际CPU凭据，复用未改Native资格；不重跑不相关验证。

证据：r/new-wave-state-root-r9374-a.json；r/P0-count-Deep-root-r9378-a.json（覆盖旧Deep RUNNING）；r/capacity-state-root-r9368-a.json；r/agent-resources-root-r9376.json；v/glm-critical-retained-export-r137c/G119A-all16-final-r137c.md；三P0冻结manifest；三份同SHA d3bc490474e767564d0b77ccdc5b39f4d2776c48574de4493666901b61fd0b58 bridge；GLM probe/model_entry/preweight；Kimi target_protocol/native_ar_decode/target_model/codec；Deep bootstrap/worker/preweight；原7ceb inference_server/agent_protocol及既有client。r=项目目录，v=本报告目录。
