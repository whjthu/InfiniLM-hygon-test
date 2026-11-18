# USGS T04 dev10 三臂效果候选

仅固定开发集10题/2图：实际original HF OneVision、同一common400/Swift模板、单A100底座加载一次，经原生AdapterRequest选择initial504、domain25、generic25。训练/正式17项/盲测资格均不继承。

原dev10是user/image-only、labels=None。实际推理按同样nativeprocessor完整编码一次/题，并在原data_collator观察处逐字段对照旧dev10 CPU身份；不重做旧CPU资格、不添加FINAL提示、不截短输入。真实encode与collator耗时单列，推理不是25步训练。

预测前固定greedy、batch1、cap256（覆盖最多50来源事件的完整Edd集合短答案，不读取gold选预算）；原template stopwords不改。观察原get_generate_ids保留EOS，即使pad==EOS亦不被engine过滤丢失。只接受唯一末EOS且未到cap；单凭stop、重复EOS/padding、cap都不能计正确。实际504权重逐臂与安全读取的原状态核对，初始状态来自原initial.pt并派生到新私有LoRA目录。

推理worker无labels参数、不导入scorer。三臂30个实际输出全部关闭/持久化/SHA封存后，另进程score_after才读取dev labels；若预测失败，保各10分母、已尝试/真实返回/缺失，直接记录未评分，不开gold。评分复用原score_once.py；三臂仅移除最多一层精确FINAL:前缀，裸合法答案同样接受，解释/多行/重复prefix拒绝，另报strictFINAL率。无预测、raw IDs/图片/金标入Git。

原domain25训练仍FAIL；其saved adapter独立CPU合格不洗原FAIL。generic25现有原8rank/800/25实际终态及14文件pin，saved==live是原冻结producer+receipt条件证据；本次仅独立读取/核adapter，不冒optimizer独审或恢复资格。

入口为infer_dev10.py，必须由root新namespace下的合格RSS-v2 bounded supervisor接入；本包不提交Slurm或自行连远端。root需发布/采纳刷新PLAN与CLAIM两对GO并重封MF；安装时核原400/15loader providers/Swift当前5源+原7源/14modelmetadata，fresh PCI/KFD/PID/port及同epoch admission。预期whole<=900、native<=820、supervisor<=850、cleanup<=880、alarm<=890、client<=895；资源1A100/4CPU(排除0/1)/RSS sampled64GiB。controller/独立terminal reader由root接入后才可运行，当前所有GO=false。

未来worker命令（占位均需root实际生成，不可直接跑）：
```text
python -B infer_dev10.py --framework ORIGINAL400 --data ORIGINAL_NATIVE_SCOPE/data --identities ORIGINAL_NATIVE_SCOPE/run/native/native-identities-private.json --admission NEW_CONTROLLER/admission.json --admission-bytes ACTUAL --admission-sha ACTUAL --epoch ORIGINAL_EPOCH --deadline SAME_ORIGINAL_DEADLINE --output NEW_EXCLUSIVE_NATIVE_DIRECTORY
```
其后以实际generation-finalized bytes/SHA调用score_after.py --run NEW_EXCLUSIVE_NATIVE_DIRECTORY --dataset ORIGINAL_LOCAL_OR_REMOTE_DATASET --finalized-bytes ACTUAL --finalized-sha ACTUAL --output NEW_PRIVATE_RESULT。dataset可L或R精确原件，禁止硬编码假设gold已在A100。
