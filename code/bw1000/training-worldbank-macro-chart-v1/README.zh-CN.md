# WDI 跨国家宏观图表数据候选

本包仅准备数据采集/派生源码。root_GO、Dataset_GO、training_GO、GPU_GO 均 false；实际数据、图、标签、训练、效果均未资格。它补充财政领域的一种跨国家场景，不产生第三条模型优化路线，不替代或解除旧 Treasury53 审批，不修改原160/108、Flickr16、VGGT 或正式17用例。

PLAN 固定8个人工地理分层，每层3国按公开 seed 的 SHA 排序各分到 train/dev/blind 一国。每拆分8国、3主题、一个7年窗口：train2000–2006、dev2007–2013、blind2014–2020。国家、年份、country-series/window 全部跨拆分不重叠；同一图的9问保持整组。这是同时改变国家与时期的领域实验，无法据此分辨两个因素的单独贡献，也不声称全球代表性。收入层次仅在实际国家元数据中统计，不用结果挑国家。

计划504个事实槽、72张512×512图、648问（每拆分24图/216问；每主题24图/216问；lookup/change/extrema各216问）。所有缺值原样保留；任一7点窗口缺值则该图9槽全部失败、不插值、不补位。相同7点显示值向量全成员拒绝；源键/日期重复和PNG内容碰撞拒绝。现阶段以上均为计划数，不是实采计数。

问题基于 ROUND_HALF_UP 两位显示数字计算，差值单位为百分点；最大值年份并列选最早。inputs 只有 user prompt+图引用，绝不含 assistant/gold；train gold另文件，未来 native SFT 接入只能显式联结训练拆分。dev/blind gold仅供隔离评分器，不部署给模型；操作人员不得查看 blind 内容挑样或调参。`score_once.py` 固定每拆分216全分母，未知/重复ID拒绝，missing/reject/cap/parse/wrong/datafail均保留。每条只有 `status=COMPLETED`、严格布尔 `natural_EOS=true` 和 `cap_hit=false` 才可算正确；缺项、整数/字符串假布尔、非自然EOS、cap和失败均记为不合格输出，报告逐项原因，不从分母剔除。只算格式严格的数值/年份 exact match，按国家整组和题型汇总。后续必须用固定同预算 base/generic/domain 三臂和相同生成约束，开发后冻结盲评，不将本分数混入正式 TextVQA/COCO。

`execute_once.py --publication <root发布commit> --claim-sha256 <CLAIM原SHA> --root-go` 是唯一未来入口，必须在根发布和依赖审阅后调用一次。先 own source exact Git bytes/清单/dirty/index 核，再 exclusive local-only namespace；单原epoch300秒含检查、7次GET、绘图与结果，Windows Job 2CPU/2GiB进程和Job commit限额，RSS另采。每endpoint仅一次 normal TLS official api.worldbank.org GET；不重定向/重试/替代源/续时。头部拒绝非JSON、压缩、重复或超限长度；HTTP体累计cap6,946,816B（≤8MiB），PNG每张≤384KiB、总≤32MiB。失败保504/72/648分母、实际请求与进度；UNKNOWN不重调。运行目录固定 `local-only/training-worldbank-macro-chart-v1-original`；raw/derived/labels 只留忽略目录，不入Git。

当前明确阻塞：固定本机解释器没有 Matplotlib。worker在任何GET前硬拦，不安装任何包。需根先冻结独立 renderer runtime，再审/pub/一次实际数据准备；实际CSV/API完整性、图像可读性、native processor完整4096模板/标签mask、首步数值、25步/新进程恢复、三轮性能、领域效果/通用能力对照和独立盲评全部未通过。兼容方向为统一框架 Qwen 原生 VLM 插件/collator 的新领域recipe，不能覆盖T01冻结registry/launcher。没有模型weights/Runtime/Torch/Slurm或checkpoint动作。
