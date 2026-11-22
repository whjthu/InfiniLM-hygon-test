独立 Gemma dev36 三臂效果源码。BASE 是实际 `.851` 目标自生 e6b624… LoRA 初态，非“没有LoRA的公开base”、不读旧受限A100初态；domain25 是原 `.851`，generic25 是本次 `.1074`/独审 `.1080`。每臂36，108原始预测全封存且独审后才首次读dev gold。当前GPU/READ/SCORE/root全部false，本机无Torch/模型/真实数据内容读取，target0。

实际前提：原 domain 独审7528B/ba9e7917…；原 domain checkpoint25 manifest2977B/897b5aad…；新generic独审8507B/261fc4eb…在 `R/results/training-gemma-generic64-control25-supervisor-v1/independent-terminal.json`；checkpoint25 manifest2977B/0e484f8e…在同supervisor目录，张量在ASSET/continuous/train/checkpoint-25。ASSET/control-independent-terminal.json 别名raw未核，不用它。原.860恢复数值FAIL不洗；本评估不训练、不恢复训练RNG。

1. Root审查采用/发布后，0GPU元数据入口 `python bind_ready.py --output NEW_PRIVATE_BINDING.json`。它在固定原路径精确读取两独审原件、两checkpoint25 manifest、.851 rank0初始化JSON和已资格396/runtime/新stage/model16 stat身份、既有dev36 stage；不读PT/模型体/图像/问题/gold。真实输出记录实际初态文件pin、两arm精确路径/14成员、runtime和模型16身份。所有数据都取真实原件，不生成generic路径/成功pin猜测，不把Slurm producerC0单独当独审。

Root应在bind前采用并封一次root/GPU/READ/SCORE执行位，训练/Model/formal/READY仍false，然后源不再变化；执行位允许此分阶段scope，不代表效果已通过。若修改源MF，必须重新生成其绑定metadata，不能混用旧binding。Windows本机lstat与fstat的ctime实际语义不同：跨API核dev/inode/size/mtime，各API自身前后仍核全部五字段；Linux跨API仍全部五字段严格相等。所有平台均全SHA，ctime变化不得吞掉。

2. Root应接既有Gemma/Swift held-tree/RSS/Slurm/新鲜PCI/KFD及独立post的独立900s lease；本包提供worker，**未新造一个已实测operational supervisor，也不自行Slurm/SSH**。可调用：
`python infer_dev36.py --binding BINDING --binding-bytes N --binding-sha SHA --admission NEW_ADMISSION --admission-bytes N --admission-sha SHA --output R/results/training-gemma-generic64-dev36-effect-v1/generated --epoch ORIGINAL --deadline ORIGINAL+900`。
admission必须有实际request/source_manifest/epoch/deadline、host/boot_id/monotonic_ns、actual_visible_devices=1、fresh_exact_GPU_PCI_KFD_port_scope_satisfied=true；同host观测<=45s。device独占/8卡父分配成本不能凭单visible改成1卡性能。原target模型16 stat五字段按同compute/boot验证，完整396源和原runtime版本/四个Swift infer provider SHA先核再import。没有默认GPU或目标连接。

3. Gemma原生入口 `get_model_processor(model_type='gemma3_vision', load_model=True, BF16, SDPA)`，`get_template(gemma3_vision,max_length=512,max_pixels=262144,padding_side=right,truncation_strategy=raise)`；是原Gemma模板和原512设置，不继承Qwen4096。实际dev36的新user-only编码计成本，任何完整输入超512即诚实失败，禁止截短或喂assistant/gold。已实测的7文件/247810B dev36直接用 `R/results/training-qwen-usgs-multievent36-effect-inputs-v1`，不复制/重render/重GET；512²PNG保持，6chart/2closure组/6类每类6。每次collator必须labelsNone，三arm同36个完整字段hash一致。

4. 每臂冻结greedy256/batch1，eval/nooptimizer，各臂重置评估seed42（不改训练初态）。一次模型load，原真实476×29,802,496参数safeCPU初态构建BASE adapter；两原checkpoint只读SHA和原生safetensors读取。三arm首次生成前逐个实际476张量比对，要求同base对象。现Swift4.4.2已有精确源码证明AdapterRequest在template.generate前动态载入；仍须目标实测该新Gemma inference路由，源准备不冒资格。raw get_generate_ids在padding过滤前记录EOS；cap256即使最后EOS也记cap失败，typed natural_EOS/cap，无EOS不得正确。每条实际返回先私有落盘，错误保planned36/108及实际attempt/completed/missing，不让failed0冒未尝试。

5. 根已有独立operational终态消费应写固定 `R/results/training-gemma-generic64-dev36-effect-v1/operational-independent-terminal.json`，字段request/source_manifest/epoch/deadline、accounting_C0/owned_cleanup_passed/post_scope_satisfied/strong_same_host_boot_owner_terminal=true、dev_gold_read=false、generation_finalized精确pin。本包 `read_terminal.py --operational PATH --operational-bytes N --operational-sha SHA --output NEW_PRIVATE_TERMINAL` 只核此实际原件及4个封存预测文件，108/顺序/actual476/原始EOS重算；不query新GPU/不读gold/不把FD覆盖说allFDclean。该接口尚须根operational producer真实接线，不是空填布尔可获得资格。

6. 独审实际PASS后，原冻结本机dataset路径下用 `python score_after.py --independent ACTUAL_TERMINAL --independent-bytes N --independent-sha SHA --predictions PRIVATE_EXACT4 --dataset G/local-only/training-USGS-multievent-render144-v2-original/dataset --output NEW_EXCLUSIVE_SCORE`。它先核独审+全部4pin/全108后创建独占first-score意图，再首次读唯一dev标签24409B/419520b0…。严格原frozen scorer/response_protocol，单列strictFINAL、contentparse/完整36正确率、cap/noEOS/missing/failed、domain-vs-base及domain-vs-generic全36配对、bothParsed、6kind与2closure组。raw文本与标签始终私有。该开发集经首次评分即EXPOSED，不冒新blind；blind36未读取。

COCO64原caption保持，与USGS FINAL目标确有格式/任务/实际token曝光差异；同初态/同25更新不等于等token/等任务，也不凭格式率晋级。许可沿原已批准COCO64内部参考（不扩张图片再分发权）与USGS政府公开数值派生图；官方来源 https://earthquake.usgs.gov/fdsnws/event/1/ ，不将派生图冒USGS官方基准。原正式17、Model、性能及领域能力均不晋级。

固定32hex：generation4d6aeb103c484796b740d4d3e4afe7ac；terminald3bade6cc7d9415ba5c1bcae777dfe1e；score91c4c857ab694548930643a4f37849e9。Root维护队列/共享状态和串行发布，本包不得重复原scope。
