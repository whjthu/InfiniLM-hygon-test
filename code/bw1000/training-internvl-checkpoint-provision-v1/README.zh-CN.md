# InternVL 单 case 原始字节供给候选

仅 T03：原 A100 17 个模型文件（含 4 完整 safetensors shard）、8 个逐 rank 的 initial.pt 和已独审的 25,357 B 输入指纹文档。PLAN 逐文件固定原路径、字节数和完整 SHA；不包含 LLaVA、数据图片或未获准的其他域。

原 A100 文件共 18,471,287,442 B，另注入已在本机持有的小指纹文档，传输分母为 26 文件、18,471,312,799 B。原 source FD 每个只顺序读一次，完整 SHA 与原 pins 比较；4 shard 同一次读同时校验原 header。目标对入流逐字节 hash，检查完整写入、fsync、稳定 FD/大小后在新自有 namespace 内 rename。目标结果是完整入流及成功落盘收据，不冒充额外重读磁盘主体或数值加载资格。

两个既有 ordinary SSH 连接分别由本机启动，receiver READY 后才启动 source；本机仅 1 MiB 有界块转发，不写任何权重。目标新自有位置为 PLAN.target_namespace；这不是对其他存储位置存在性的判断。两个端点固定 2 CPU affinity、nice 10、512 MiB RLIMIT_AS（不是 cgroup RSS 硬限），纯 stdlib，不调用 Torch、SDK、Native、模型、GPU 或新 Slurm。CPU 与 RSS 成本使用实际 rusage，物理磁盘/网络 IO 未测。

一次 origin epoch+21600 同时约束本机、两个 endpoint 的 SIGALRM/monotonic tick、进度和退出；不在握手、文件边界或重连时重设。新 source/receiver/local namespace 都 exclusive create，任何旧 namespace/partial 拒绝覆盖。source/receiver 当前原件、进度、已完成和失败分母保留。强杀本地 SSH、255、过期或未获 READY 一律远端 UNKNOWN，不重连、不重播、不自动清理或重命名。

发布前本地检查（无 SSH）：

`python -B -X utf8 verify_package.py`

`python -B -X utf8 check_CPU.py`

root 普通发布后唯一传输 CLI：

`python -B -X utf8 relay_once.py --publication <published40hex> --source-manifest-sha <SOURCE_FILES_SHA256>`

root 必须从固定 Python 执行上述入口；入口用 git show 比对本次发布的 MF、全部递归源、claim 和既有公开依赖，再做实际发布准入。local 25 KB 指纹只能按原 SHA 注入，不进 Git。不得改用旧 transfer label 或重放 UNKNOWN。

传输返回或 UNKNOWN 后，root review 后可唯一执行 `python -B -X utf8 reconcile_original.py`。它从原本机 intent 恢复 context，一次 90 秒只读两个原 namespace 的小 intent/progress/outcome、原 PID/startticks 和目标 26 项 stat。没有 namespace/没有终态仍 UNKNOWN；不扫描 /proc，不读源权重，不重新 hash 17 GB 目标。真实 receipt 与 PID 退出闭合后才分别判终态；目标 SHA 是原收到的全流 SHA+落盘收据口径。此项与后续完整模型数值、训练、恢复、性能、效果验收分开，GPU_GO/Model_GO 均 false。

intent/READY/outcome 都绑定原 hostname/boot/UID/PID/startticks。只读观察落到其他 login HOST 时，只核共享存储原件；原进程状态为 UNKNOWN，不能用不同 boot 或其他主机没有该 PID 当原进程已退出。不提供新 second-hop/fallback，不因此放行重传或发布模型资格。
