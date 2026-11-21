# T09 v3：严格4/5字段ABI兼容，原缓存25次原生更新

原v2 .762 pre / .763 GPU FAILED2:0（125s、0/25、empty steps）/ .764 post及独立释放保留。源码定位确认文本refiner会走attention默认flash，外层set_attn_mode(torch)并未覆盖该调用。实际.791 CPU COMPLETED0、7s、0GRES，当前9535B/31832e… provider返回4，finite/first4/roundtrip均通过；原1594B结果SHA63abd095…作为新scope强前提。该证据支持ABI兼容，不冒已获得原失败完整栈或原A100数值资格。

新request df2a0db37f494c0a9f149260d71cb5e6，RUN=ROOT/results/training-hunyuanvideo-cached-native25-target-v3，仍1087656/f11r4n17。root/GPU/training/READ四执行权限true按root明确授权，资格flags全false。一次25实际更新=5次计时预热+20计量，0恢复；原普通common CLI、Muon/LoRA/GC/BF16、seed、输入/cache、完整模型、checkpoint10/20/25、1440/1240/1200+post预算、accounting/KFD/held cleanup/失响应不重播均不改。新epoch/preflight必须绑定新request，不复用v2意图或输出。

唯一原生差异：原3909B e9835a76… helper中4个unpad_input赋值，删除从未读取的第五接收变量，并经私有 _t09_unpad_first_four 包装。仅接受内建tuple/list且len恰4或5；返回values[0..3]的原对象引用，不reshape/转换dtype/替换值，也不切换attention backend。FA2/FA3原kernel调用、参数及所有数学AST保持；五轴shape解包不动。去掉新增helper并逆还原四处赋值后，整个模块AST必须等于原件。

flash_compat.patch_bytes硬绑定原SHA，生成4212B/f9762977bf93bf7ea41258b32eafa6985b9461702274882b42cb71d112e084a0，保原Tencent许可证和其它原字节；派生正文仅private/target，不进Git。COMPATIBILITY.json仅小hash合同。HOST controller在已保存本次intent后生成RUN/overlay/flash_attn_no_pad.py，新文件只写一次再chmod444。原source全38成员仍按26f1c4…核，原helper e983…从不写、不链接、不更改。

实际Apptainer保持原source目录ro bind到/t09-native/hunyuanvideo-1.5，额外仅将新overlay文件只读bind到其hyvideo/utils/flash_attn_no_pad.py。GPUworker核.791原receipt确切1594B/SHA、四字段/roundtrip语义、用同active DTK PathFinder解析当前provider并核9535B/SHA31832e…，不重做CPU函数。实际alias和backing需相同完整stat，alias statvfs须ST_RDONLY，并在实际容器Python compile新正文；canonical原helper仍e983…。这些门禁通过后才进入原native CLI。没有猜runtime/site或先import模型。

HOST54/compute49两个完整模型fingerprint与原.733 fullSHA收据继续严格区分，绝不忽略dev。HOST准备overlay先只解码核.791 receipt结构；PathFinder及实际provider字节/ro挂载只在GPU容器核。consumer在原Slurm终态后只读原记录、三checkpoint、provider/overlay证据和独立post；不重模型、不把资源clean洗成训练PASS。原first-step replay/UNKNOWN transfer/正式17/域效果均独立保留。

controller调用仍为 python -B RUN/src/controller.py --workspace RUN --epoch ORIGINAL_EPOCH --source-manifest-sha256 FINAL_MF --preflight-sha256 EXACT_PREFLIGHT_SHA。root须先发布、重新fresh准入后唯一调用。无默认SSH或下载器。

本机check_compat_CPU以原helper真实AST/源码函数循环执行FA2、FA3，4/5返回均同first4、invalid3/6在kernel前拒绝；kernel显式CPU stub，不能冒真实GPU attention数值。真实.791 CPU provider证据不重跑。原首次-I本机harness隐藏scriptdir导致0tests的失败保留；修正为与实际worker相同-B后5项通过。新增receipt gate另窄测；最终真实package ingest校验确保operative flags/MF接线，旧native fixture不重跑。原始provider和patch正文留local-only。
