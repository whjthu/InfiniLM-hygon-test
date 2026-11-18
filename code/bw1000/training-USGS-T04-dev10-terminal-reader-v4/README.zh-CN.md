# dev10 v4 独立原终态 reader（准备态）

新 exact namespace 仅指向 training-USGS-T04-dev10-effect-A100-launch-v4。复用已发布reader-v3的terminal-first、原source/claim/epoch、真实owned PID/startticks同host/boot退出、完整3臂×10原预测与原native return、rawEOS/cap核验。旧v1/v2/v3失败不覆写、不重读。

唯一观察合同变化：主GPU准入由UUID/PCI/actualminor、无compute-app、38000/40960MiB、端口与已观察设备FD冲突检查支持。FD为辅助覆盖，PermissionError或显式进程/FD消失可标PARTIAL_UNKNOWN，原PID退出不从消失事件推定。reader逐记录核事件类型/阶段/可得身份与变化拒绝；incomplete_total是事件数，incomplete_processes是唯一PID数，completed+partial_process_attempts=attempted。最多64条/default sorted ASCII JSON≤16KiB，截断后保total/truncated分母；不独立重扫描/probe，不把覆盖UNKNOWN变clean。

post/outcome仅scope release；exe UNKNOWN、allFDclean/exclusiveGPU=false。实际可读selectedGPU/KFD持有者必须为空。root负责单次发布/采纳/调用，READ_GO默认false、真实request/publication/epoch/MF/claim待原once后填独占private context。不能用null启动或重播旧scope。

collector为2CPU排除0/1，原60秒含安装/传输；raw小件合计6MiB/private bundle8MiB，每普通JSON256KiB、原return64KiB。0Torch/SDK/device查询/gold/权重或状态张量。before-held-Popen失败若attempted=true但缺child锚保UNKNOWN；只有control false+原未release且supervisor真实退出才能独审无child准入失败，并保持30 UNOBSERVED。

Root通过既有ordinary SSH注入 CONTEXT/READER_CLAIM/READ_DEADLINE到collector，private stdout留L/R。离线调用：`python -B read_original.py --bundle <privatebundle> --bundle-sha256 <actualSHA> --context <actualprivatecontext> --output <new-exclusive-private-result>`。不评分；完整30封存后的gold评分由另授权score_after执行。

本次只源码/AST，未运行collector、未重复旧fixtures。全部GO false，待root审阅。
