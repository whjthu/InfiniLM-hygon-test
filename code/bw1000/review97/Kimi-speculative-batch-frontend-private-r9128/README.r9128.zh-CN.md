# r9128 原生批量投机 Python 前端交付

首版 source freeze，唯一 native/Model launcher owner 为 r9122b。runtime8文件见 producer index；copy 到其 runtime 同目录，target_binding/startup 由 nativeowner 提供。保持原714/1048和r9116源不变。

CPU实际四节点线程+真实loopback二进制socket、真实ctypes数组布局完整12cohort PASS，数学mock，不是GPU性能或实际C16/32资格。复用真实ContextStore进行请求context隔离，32key错epoch/错past/独立retire验证；mock原生内部lengthledger与outerPair分开，部分重放后下一forward实际检查，异prompt/past/末宽、混合EOS、I192OFFON、append/stop/容量拒绝均覆盖。CPUfixture给odd请求cap8以强制ragged，非最终性能配置。

当前原生能力严格读取implemented_max_requests8/flatten_tokens16/pool17/flags3，以及batchtapcap0x01080500；不将C16/32假称已支持。首矩阵C1AR16token热身、C2W1AR/C2W8oldspec/C2W8stateOFFONONOFF、C4W1AR/C4W4oldspec/C4W4stateOFFON，共11cohort，GSM0/1不同prompt轮换、cap256。closed cohort，实际EOS只退出本用户，不复用槽；不是连续到达服务。AR为原生batch W1路径，非旧ordinaryAPI。

每cohort reset后显式batch_mode(owner Clear)，之后再prefill；nativeworker RESET也清理，但显式协议不依赖隐式缓存寿命。每次I192在所有node统一case前设置GROUPED_R8669Y/EXECUTION_R4665为0或1并barrier，workspace/compact/Math0保持原样；记录每node实际8×2前后计数，不能仅看配置声称T16分支已执行。

新草稿batch_draft_worker_r9128.py，GPU1，report Kimi-batch-draft-r9128-<job.step>.json，status ready/COMPLETED与normal_close，末remaining_request_keys_before_close应空。所有host/port/nonce沿KIMI_*_R9046；begin_cohort(epoch)只在上cohort全部retire后；query/retire均带(epoch,request)，append严格past连续。每请求GPUfeatures单独保存，权重共享；当前query逐req串行，并单列RPCgroup、RPCsum、querysum与每reqserial_queue_wait，后续C16/32需真正draft微批/多卡Pareto，不能把mutex或串行RPC称为batchdraft。

report batch_cases→request_metrics保存实际每用户first/last/common native clock、(N-1)/(last-first)、含尾处理的保守decode_user、nativeTTFT/ITLp95/p99、EOS/截断/quality、所有tokenIDs；batch cycles保存真实activeC/raggedwidth/T及全周期。总decodecommon window从最后prefill首token就绪至全部用户结束，包含draft/capture/传输/verify/replay/commit/retire；全32target+1draft=33卡分母。tokens包含thinking/channel/control，非可见HTTP指标。prefill EOS的原生行仍phase1但前端永久退出，不在同cohort复用。

受控JSON队列 env KIMI_BATCH_CASE_QUEUE_R9128，建议basename Kimi-batch-queue-r9122b，路径/refactor/results/<basename>/q*.json。先写tmp再原子rename。case示例 {schema:1,action:case,id:unique,C:2,W:8,strategy:state,I192:true,cap:256}；可选requests长度C，每个{question,answer,sample_index}，answer仅scorer metadata，不放prompt。cap<=512、prompt<=512且sum<=1024；C<=8、C*W<=16，非法容量记command_rejections不截题；cohort完成才reset。stop为{schema:1,action:stop}。无任意Python/代码加载/SDK热换。

budget沿加载前experiment_started_r9046+5000，每case前至少180秒余量，while有全局deadline，外层5200inner清理。队列调用方应为cap512长case预估充分预算，不在最后180秒强塞最慢case。所有模型级质量和性能等待真实Model/audit/normal/post，12case CPU通过不证明目标已达。
