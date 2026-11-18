# USGS 新事件确认组一次数据资格卡

现有公开独审证明原50事件全部覆盖一次，16图80QA；没有未消费的可复用独立组。本卡不读取旧训练/开发/已暴露blind问题、gold或预测，也不重新训练/选checkpoint。原initial/domain25/generic25、25更新、cap256和原严格显示数值评分器保持。

先固定新官方历史查询（2026年9月、M>=4.5、earthquake、time-asc、first200）再一次GET，<=1MiB，正常TLS，无HEAD/count/redirect/retry/代理/换日期。本地总新数据<=20MiB。旧50只读原source-records-private.json精确23081B/dd3622...，新旧一起按aliases、原30度纬度/60度经度格、72小时500km近邻做传递闭包。凡连到任一旧train/dev/blind事件的整新component全排除；不把粗区域等同绝对无泄漏证明，目录不全/别名不全/相关性仍是限制。

按固定seed+组ID哈希顺序，组内事件时间/ID排序，分原<=5事件非重叠窗口，组间roundrobin取恰好5窗口且至少3新组；不足即FAIL，不换参数、不切碎窗口补位、不看答案选择。分组/窗口declaration先写盘，随后才同原5类问题/Decimal2显示舍入产25gold和答题inputs；新confirmation是独立确认拆分，不搬旧blind进train。图是官方数值本地新渲染，不是官网原图；官方许可和源网络出处见SOURCE_PROOF/CREDITS。

当前所有GO=false，0新sourceGET/渲染/native/model。root审查采用root/CPU/source_GET三GO、普通发布后唯一固定Python执行：`with_qualified_overlay.py -- --publication ACTUAL --claim-sha256 ACTUAL --root-go`。120秒从wrapper开始，含qualifiedoverlay/发布核验/Git/oldsource读取/WindowsJob/GET/closure/render；现有实际qualifiedMatplotlib环境/font硬门禁先于GET，2CPU/2GiB Job进程及总commit硬限制，RSS只OS采样。已有RUN/未知执行均不能重播。

实际资格若成功仍只DATA；独立审完整新source/closure/25gold/5PNG与有限视觉可读性后，另新300秒0GPU nativeCPU（原统一T04单user/labelsNone/完整4096和字段identity/14meta7Swift3wrapper）才可产生新native25资格。之后再独立75预测全seal→实际terminal→首次gold。当前不调用旧native或GPU、无训练loop/daemon/定时器。
