# USGS dev10 A100 独立 launch-v2

v1真实失败保留：pub6949a3ef/结果f374c88a，原guard只知BOUNDED_NVIDIA_QUERY_FAILED、未释放native child；原processidentity/devicepost仍UNKNOWN。之后独立只读诊断发现minor_number不受支持；不把新诊断的rc/日志补写进旧guard，不把旧失败当清场证明。

本v2独立namespace与一次request。仅修被动准入：SMI使用实证支持的5字段；实际/proc/driver/nvidia/gpus/<canonical4位PCI>/information核GPU UUID、Bus Location、Device Minor三字段各一次。minor来自该字段，绝不取GPU index。原282B schema和5字段rc0证据的路径/SHA在SOURCE_PROOF。

controller在任何GPU query前持久保存owner/control-state；installer保存PID/PPID/PGID/startticks/UID/host/boot。primary与post query失败各自保存code、rc、完整捕获日志的bytes/SHA和最多1024B私有预览；post失败仍UNKNOWN，不冒clean。

冻结18core数学/FINAL/scorer、原RSS-v2四源码、单base/三LoRA30输出、900 epoch与held/tree cleanup不改。真实A10040GB total40960/free>=38000仅预/后准入；只选固定GPU0 UUID/PCI，完整ownUID selected-minor/KFD FD、computePID、port需fresh核；任何占用/未知失败，不杀既有owner、不改选卡。GO前仍同父monotonic<=60s。

新native目录必须初始不存在。8个GO均false；root采纳后seal.py --adopt→verify→普通pub，再唯一固定localPython -B -X utf8 submit_once.py --publication ACTUALPUB --root-go。source校验/Git/上传均含submit第一行原epoch900；native820/stop850/cleanup880/alarm890/client895不重置。SSH unknown与任何存在的本地/远端exactnamespace仅reconcile，不重播。

A100不部署或读取gold，不运行scorer。30实际预测sealed+独立terminal/post后，root另scope有界取回4精确private输出，在L原dataset用core/score_after.py；预测未封存前devgold不开，blind始终不开。

当前只是source：18项新字段/parser/identity边界一次通过；不重复core15/RSS17/v1九项，不新远端/GPU/模型/权重/Git。v1所有源和原件不改。
