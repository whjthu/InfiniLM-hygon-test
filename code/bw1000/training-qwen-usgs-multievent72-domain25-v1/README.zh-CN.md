独立 Qwen 多事件图表配方：已在目标的同一72条训练问答/12PNG进入Qwen原生模板新资格，再做连续25次真实优化器更新，最后比较原公开base与本次saved25。不是旧88条/.501/.544重跑、不是Gemma配对、不是VGGT .927/.978，也不把任何终态重新放回READY。

训练数据来自已有USGS render144-v2，原711个官方事件快照、旧750条元数据别名+72h/500km闭包排除、新coarse30×60空间closure按组先split后QA。固定train72/dev36/blind36；本包不打开blind任何正文，不把开发集改称fresh blind。新72训练覆盖12张六事件图，3个训练closure组；开发集6张/2组。它们不是72个独立事件窗口，800 modulo72曝光更不是800个独立样本。未按模型效果/答案选择或更换数据。

目标已有训练数据根：`R/results/training-gemma-usgs-render72-native-CPU-v1/data`，原72 JSONL54246B/a897e7…与12PNG已在Gemma .511/.845真实核过。那不是Qwen处理器资格。新任务先以0-GRES/三mask=-1、10metadata-only模型视图，完整4096/raise而非截断、原Qwen模板/真实supervised suffix含EOS、全部72 encode+collator硬门禁；失败不进入GPU，保72/25分母，Torch隐式GPU初始化仍UNKNOWN不伪0。

Qwen完整14组件（含4权重分片）与原初始LoRA复用原已准入路径，不下载或传权重：`R/results/training-qwen-original-public-model-v1/model`、`R/results/training-qwen-initial-trainables-v1`，原base393框架在`R/results/training-qwen-a68952c-swift-initial-CPU-v1/framework`。本次HOST重新逐文件SHA/稳定stat、只复制393到新RUN；数学/通用translator不改。训练保T01原8卡/BF16/bs1/acc4/SDPA/语言GC/r16a32/dropout.05/AdamW/cosine/warmup、原504张量43,646,976初始值，checkpoint10/20/25各14成员。原初值只恢复trainables，不称RNG/checkpoint完整恢复；本包不运行任何resume。

确切新请求`c933ba7359f84e8b9b80ac40d2d30964`，1103606/f13r1n10（必须root fresh queue/owner/active-step/8PCI/KFD/port29941核验，无旧UNKNOWN或ownerhold才运行）。whole2400含本机源/Git/运输与HOST准备；CPU≤300/GPU≤1800/native≤1700，accounting/post留原预留。源全部GO=false、READY=false；root采用刷新CLAIM/MF后经现有授权transport调用`launch.main(publication, transport)`，无默认SSH、无自动重播。

开发输入补齐是精确7个既有文件247810B：36user-only JSONL+6PNG。已本机实核完整解码，私有331599B bundle和receipt在`local-only/training-qwen-usgs-multievent72-dev36-prep-v1-original/`。根已实际一次落位并逐件重核`R/results/training-qwen-usgs-multievent36-effect-inputs-v1`；`stage-complete.json`968B/SHA9411e7c43788145a56e3fe1ca20cd163bf0b0f9329ff3f3642a9a8cffec3dd61。controller训练前严格核该收据及原7files，gold未上传。这个目标数据缺口已闭合；Qwen新72原生资格和本次设备鲜准入仍须实际通过，不凭本机验证记READY。

25完成/all8 receipts/三checkpoint/训练owned-tree退出后，同一个8卡租约内另开一个仅可见HIP0的推理子进程，base加载一次、再接本次saved25 LoRA；实际依然记8卡分配成本，不能冒单卡perf。实际逐参数torch.equal核激活adapter504；若原初值并非全252个LoRA-B为0，不允许把公开base冒相同初值，保25更新并拒绝该效果比较。全72预测/完整user-only输入hash/自然EOS raw IDs/cap256先封存，再经独立`read_terminal.py`核原件和post，才允许本机`score_after.py --independent ... --independent-sha ... --predictions FETCHED3 --dataset ORIGINAL --output NEW`首次读dev36 gold。原response_protocol/frozen_score/contracts逐字节复用，全36错/parse/cap/missing保分母、报strictFINAL及配对win/loss/tie；blind36始终封闭。root负责精确3文件获取，agent未执行任何目标操作或评分。

许可与来源：USGS ComCat官方数值政府公开数据，https://earthquake.usgs.gov/fdsnws/event/1/ 和 https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits 。保USGS归属、幅度类型仅比较reported NUMBER/非能量风险、两位显示值及完整ties。图表和问答是本项目衍生，不冒USGS官方图/原formal17 benchmark。源码/小hash清单可发布，原图片/问题/labels/权重/预测留私有目录。新模型效果、吞吐、正式17和Model资格都保持false。
