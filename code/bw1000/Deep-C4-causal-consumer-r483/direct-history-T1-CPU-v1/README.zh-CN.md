# DeepSeek C4 单 token 历史 operand 编译

同一用户的单个 token 在正的压缩历史长度且本步没有追加条目时，直接将既有 `PagedIndexerCache::gather` 返回的 key/scale 交给 `DeepseekV4IndexerScores`。原分支重新分配并填充 key/scale，再复制同样的历史。多请求、空历史、追加新槽、非 T1 均走原分支。保留聚合分数、TopK、状态检查、持有对象及同步层级。

| 项目 | 原接口 |
|---|---|
| 历史 key | UInt8 `[W,128]`，连续，元素 stride `[128,1]` |
| 历史 scale | FP32 `[W,1]`，连续，元素 stride `[1,1]` |
| 调用方 | 同一个 DeepSeek C4 score provider，原 `libinfiniops.so` SHA `e1d9432e6de79cf09782fc3b3b332fb969d1a948dbac513b3da677d78f7f1f09` |
| 唯一变量 | OFF 保留填充和复制，ON 直接使用 gather 返回的同形 operand |
| 实际样本锚点 | 910374.1534 的 21 个原 cache 的 prefix2124 gather 包围时间 1.59–2.26 ms。该结果是旧 slot metadata 对照 |
| 源码成本推导 | W=2124 时各少 280368 B 的逻辑 fill 和 D2D payload，另省两个同大小合计 280368 B 的临时 buffer |

新包只替换 CPU1113 已验证闭包的 TU31。检查 fresh `-M` 与生产者完整 SHA 图，实际 `-MD` 复核同图，再按原顺序将新 TU31 与 49 个原对象链接。核对强动态符号、NEEDED、Bsymbolic 和实际暂存 ELF 解出的 68 条依赖身份。原对象、二进制和运行产物保留。真实编译输出为独立候选目录的 `CPU-result.json`、`TU31.o`、`libDeep_c4_direct_history_t1.so` 和编译/链接日志。只做 CPU 构建与 loader 元数据观察，SDK 和设备数值仍待测。

协调者先按项目流程发布本包和同名任务认领，然后通过现有批准的 SSH 连接调用 `launch.py` 的 `main(publication, transport)`，其中 `transport(body, stdout, stderr)` 执行 HOST 的 `/usr/local/python3.12/bin/python3 -I -S -B -`，以受限时长把 body 送入 stdin。返回的是原 srun 拥有凭据。原节点需当前无其他数值 Slurm step，且接收器验证新 label 和 namespace 未出现。连接断开后按原 intent/Slurm 句柄对账。重新分配节点须采用新认领。

后续独立 GPU 对照须首先取得原 r483 七权重范围凭据并完成 weighted 原诊断。以同样的 TP1、context 32768、16 GiB floor、checkpoint SHA 和固定 causal 日程对比新/原 DSO。逐样本保留 cache payload、FP32 scores、TopK indices、state、commit/abort/reset、正常释放以及异常分母。主比较取 past8496 的 T1 稳定配对，追加槽、T2、多请求均走原回退对照，不改 checkpoint 或数学。全部过后才讨论 Native/Model 性能。
