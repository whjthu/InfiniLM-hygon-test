# 固定盲组75预测独立终态reader

复用dev-v4已审collector/audit逻辑，新namespace固定blind25effect，3臂每25/总75。只在root确认唯一原作业终态后一次collector，读取小元数据、75native-return、3predictionfile和其SHA；不读题面/图片/gold/权重，不importTorch、不query设备、不重推理。

root_GO/READ_GO初始false。Root采用并普通发布后注入actual CONTEXT {namespace,request_id,publication,epoch,source_manifest_SHA256,claim_SHA256}、READ_DEADLINE<=now+60和published READER_CLAIM，一次运行collect_original.py，stdout只ignored私有留存。实际supervisor/child/registry必须同hostboot+savedstartticks已退出；原FAIL/postUNKNOWN不洗。

离线 read_original.py --bundle PRIVATE --bundle-sha256 ACTUAL --context PRIVATE_CONTEXT --output NEW_PRIVATE_JSON。75逐slot核实际native-return与prediction、原25sourceid/nativefieldguard、自然EOS/cap由pad过滤前原IDs重算；cap或非EOS保完整分母。三predictionfile必须与finalized实字节SHA一致。原CPU25 actual2014B资格和35142B identities硬pin，新native55/dev资格不继承。

primary SMI5 UUID/PCI/actualminor/noapps/memory/port与FD辅助覆盖分别核。FD unreadable/lifecycle仍partialUNKNOWN，已见GPU/KFDholder仍拒，不冒allFDclean/exclusiveGPU。原domain训练FAIL保持。结果给score_after的7字段契约只在75真实文件和cleanup/source元数据合格后放行；gold仍false，另本机评分进程才可首次读25gold。原sourceonly，不执行reader或fixtures。
