# InternVL official model03 单1MiB HTTPS Range可达性源码

当前仅prepare，CLAIM root/CPU/HTTP/READ全部false，GPU/Model/training/whole_download始终false；0 probe HTTP、0远端、0权重body、0fixture。原362由root已真实终态独审闭合，本包不重query/传输/reader其原scope。新label/namespace只跑这一次新CPU探针，不是原Gemma或旧denied payload范围。

唯一对象为官方OpenGVLab/InternVL3_5-8B-HF immutable819e391ba0a9eea8d9d8b1c27cd1e777b0569415/model-00003-of-00004.safetensors。root既有公开pointer4915962496B/SHA595f3b90…与冻结A100 PLAN匹配；这不是本轮重新读取完整weight证明。02只有其他revision SHA旁证，04同commit SHA已见但页面size仅2.32GB，二者exact samecommit size均未确认、不晋级。PUBLIC_EVIDENCE.json保完整primary URL、公开apache-2.0标记和确认边界。

固定GET Range: bytes=0-1048575，Accept-Encoding: identity。正常系统CA/TLS，urllib ProxyHandler({})明确禁代理；不读HF_TOKEN/netrc/密钥、不用SDK、不发Authorization/Cookie。只允许固定官方HF/Xet/CDN exact FQDN（当前HF官方host表及HF staff历史cas-bridge兼容依据均列PUBLIC_EVIDENCE）；无wildcard/S3/cloudfront任意host。最多3次redirect、最多4个HTTP GET attempt，每次保持Range/identity；无HEAD、重试、mirror、HTTP降级、auth fallback或换文件。

NoRedirect阻止urllib自动follow/drain；手动只读redirect headers，body显式消费0。每hop仅保存status、hostname、URL SHA256，Location/signed URL全体不落stdout/receipt/log/PC/Git；异常也只存安全类型/常量。最后必须206、唯一Content-Range=bytes 0-1048575/4915962496、唯一Content-Length数值1048576、identity且无Transfer-Encoding，三条件通过前不调用body.read。200或其他错误立即关闭并显式body consumed0。网络库/内核物理socket预读未知，不误称wire0。最终最多显式读取1MiB，不额外read+1试探EOF；HTTP framing/CL限定body。1MiB opaque文件与临时partial仅R compute新scope；PC只收小metadata及新observed prefix SHA。它仅证明HTTP/range网络语义，绝不证明5GB wholeSHA/权重完整/model/native/training合格。

资源固定parent1087656/n04，alloc4 CPU、worker2 CPU（排0/1）、Slurm2GiB及AS2GiB、0GRES；4allocated不等4busy。复用StageA qualified held_child和原host parent-owner/lease/node/active-step/label去重/fresh sacct口径。Slurm官方文档说明--time秒数按整分钟向上取整，因此实际请求60s而不是90s→120s；仍符合step<=90上限。worker截止min(original epoch+80,step_end-10)，host98、held清理105、producer accounting110、独立reader/fresh exact新step accounting115、普通SSH客户端118/本地回收119，全部同一个120epoch，不足就STOP。

入口launch_once.py --publication EXACT --source-manifest-sha EXACT；root独审采用CLAIM四开关、普通发布后方可唯一运行。复用既有普通SSH helper，代码小包通过一次纯源码stdin安装（没有后续stdin payload，故python -正常EOF执行）；不新增SSH fallback或service。worker在compute直接HTTPS，小body不走PC桥。run目录、worker目录及Slurm新label拒重；timeout/未知不自动再GET。

host先保原producer outcome，再实接新read_terminal.py进程：只读本次小JSON和原source MF、fresh sacct exact新step/label一次，与producer原终态核对；不读1MiB、旧receipt或原362。HTTP失败且终态可独审时分列独立terminal真/HTTP资格假，不洗失败；源码、数据、HTTP、Model资格分列。原14/initial/partial不读写、不链接、不rehash；不加载模型/Torch/SDK/GPU。当前仅AST/MF，未声称真实接口或网络已通过。
