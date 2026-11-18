# USGS 强事件关联三分数据准备 v2

仅新源码，root/CPU/source_GET/Dataset/training/GPU/Model GO全部false。0GET、0新题/gold、0fixture、0native/模型/训练/评分。未执行的v1源码及旧confirmation规则/结果原样保留；本schema在新source/题/gold全未读时预注册修订，不事后改变当前T03评分。

唯一官方查询固定2020-01整月/M>=4.5/earthquake/time-asc/limit500，单GET、<=1MiB；无HEAD/count/redirect/retry/改窗/代理。接受到达500的有界first500样本，不冒全月完整catalog；cohort-private.json记录实际首末UTC时刻、实际count及hitcap。旧50实际2026-10-01..02、旧200实际2026-09-01..10，与2020-01不重。旧250数值metadata和原2724边只读PLAN三份精确pin，不取旧HTTP、不读题/gold/预测。

新强图只使用同alias，或同时满足72小时及500km的关联；传递闭包中的任一旧250事件均为排除锚。原2724边保留原件和SHA历史证据，新图仅使用其alias/near子集，coarse-only旧边不并入新强图；新增/old-new边也不使用coarse相同作为关联。粗30x60仅统计与分层，明确允许旧/新及三split地理重叠，不是地理OOD。旧v1规则下旧250占20/36格、16空格中11在极区的先验约束不再误当事件泄漏条件。

固定分配：groupID=SHA(排序event IDs)，stratum=组内最小字典序coarse cell；先stratum排序，组内按SHA(seed+groupID),groupID排序。每个整组分配给使(当前stratum该split组数、该split总组数、SHA(seed|stratum|groupID|split)、split名)最小的train/dev/blind。完成所有组分配之后，各split按group hash顺序roundrobin取组内time/ID排序的不重叠<=5事件窗口。每split恰好10图/50QA且至少3强连组，总30图/150QA；任一不足STOP，不改阈值/换窗/交换组/补位。余train窗口只声明候选，不产生额外图/QA且不自动训练。任何新强边数超过250000、source/identity/numeric/renderer/budget失败同样STOP，保150 planned分母。

未来评估只使用dev+blind的100QA（同三冻结checkpoint共300预测），train50QA留给另行授权的持续领域训练。EVALUATION_PROTOCOL.json及response_protocol.py在GET/render/gold前随源码普通发布并SHA封存；同original initial504/domain25/generic25、同prompt、greedy/CAP256/batch1/seed42。content与strictFINAL双指标协议和parser数学主体相对未执行v1不变：只取无歧义末行；固定Decimal/km-m转换与已知图ID归一，不读gold选择候选数字，不扫描择优，不用LLM。原T03 0/10、10/10、0/10原分数保留。实际能力结论不能由基线格式失败代替。

入口原with_qualified_overlay.py -- --publication EXACT --claim-sha256 EXACT --root-go。root先独审并采用PLAN/CLAIM共6个root/CPU/source_GET字段，再普通发布才可唯一执行。原qualified overlay、JobObject 2CPU/2GiB进程及总commit硬限、120秒单epoch/8秒退出余量完整保留，含源码发布核验/GET/强图/数据生成/退出；RSS仅采样。新本地数据总<=20MiB，PNG总<=10MiB。已有namespace/执行UNKNOWN不重播。

原numeric parser、显示Decimal/5类问答数学、HTTP有界流/同固定2020URL、renderer、旧scorer、Job/时钟/overlay精确源码复用；本轮只改强图语义/预注册分层三分/30图150题路由及cohort身份记录。source complete不等dataREADY。独立source/math/视觉资格、InternVL新完整输入nativeCPU、后续训练、固定checkpoint开发及最终新blind效果均须各自新stage；预测全封存和独立terminal先于首次gold。旧T04 blind25/confirmation25与75预测、T03 dev10已暴露，不重新叫blind。本任务不是TextVQA/CIDEr/formal17验收，不给domaincapPASS。

历史源码边界：frozen_scorer/contracts.py为保持原评分器精确SHA保留，其中旧grouping函数仍含coarse-region历史规则；score_once仅导入其need/sha，不调用该grouping函数。v2数据强图只走select_groups.py，不使用该历史grouping入口。
