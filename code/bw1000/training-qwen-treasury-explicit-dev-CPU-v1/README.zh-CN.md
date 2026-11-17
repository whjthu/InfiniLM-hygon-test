# 新显式指令 dev108 的独立 CPU 准入

本包只做新开发输入原生预处理，8CPU/12288MiB/0DCU，父1087656/a01r4n14。原始dev108/36图按旧c12efcc…库存核验，恰36 signed_difference追加与训练相同的FORMAT_INSTRUCTION，其余72题、图片、ID、顺序不变；不读标签/盲测。已在本机真实构建3个metadata共60228字节，新inventory为07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8。原36图不复制。

dev_revision.py 可用 --original 原dev目录 --original-inventory 原inventory.json --output 新目录复现。根普通发布后，精确安装SOURCE_FILES清单到R/results/training-qwen-treasury-explicit-dev-CPU-v1/source，将本地新dev的3metadata安装到RUN/dev、inventory.json到RUN/inventory.json，创建空RUN/dev/media。不要在这个HOST绑定点复制图片或放其他文件。launcher在HOST逐原39files验证新metadata；容器把原R/results/training-qwen-treasury-dev-native-CPU-v1/dev/media只读嵌套挂载至RUN/dev/media，实际nativechecker会重新核39新库存及全部36图。

以绝对HOST Python执行source/launch_once.py --publication COMMIT --claim-sha256 SHA。只有一次600秒epoch，inside最多450、Slurm最多480并按原剩余减40秒缩水，inside+kill10不超过Slurm；客户端与boundedaccounting沿同wall/monotonic期限，UNKNOWN保留PID/原目录不重提。复用326成功v2的精确hyhal允许路径：/opt/hyhal严格解析/usr/local/hyhal且target非链接目录，其余项目目录继续拒链接。没有DCU和模型权重加载；原checker使用load_model=False处理器，visual/textonly各108题，目标实际资格待首次运行。

该checker与旧check_qwen_holdout_native.py唯一字节变化是库存SHA。旧289不能替代新CPU资格。CPU stdout仅版本、长度、imagecount和摘要，不含题面或答案。ROOT_AUTHORITY.json只提供根已独审328的boundedhash权威；未来GPU仍须核原targetartifact，不能重reader或把publicsummary当实际验收。

后续新GPU包仍未创建：固定base/old_domain285/domain328三fresh进程，各同一修订dev108、greedy64/BF16/SDPA/batch1/4096和同wall预算，324完整分母、每topic每臂36，保全部wrong/cap/missing/invalid，严格原评分，不抽末数。所有三臂须重新生成，不复用旧298输出。不宣布旧训练消费顺序已等预算对齐或因果胜利；不是正式质量/盲测。5步训练已结束，不续训、不做prompt/capgrid。

Slurm 会按分钟向上圆整，故本包只申请向下取整后的整分钟：60*(min(480,intremaining-40)//60)。不足60秒，或inside+kill10不落在该Slurm时限内，拒绝新step。例如剩161秒请求120秒；剩100秒请求60秒；剩80秒拒绝，不发送36秒后被向上圆整的请求。终态观察具有独立terminal_deadline，最后一次query最多只用该窗口剩余时间，耗尽返回UNKNOWN而不额外延长12秒。9组实际AST预算/argv和5组querytimeout边界已核，未重建metadata或运行native。
