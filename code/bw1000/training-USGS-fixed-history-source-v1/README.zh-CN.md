# USGS固定历史地震源：一份≤1MiB GeoJSON，未执行

实质新科学场景：地震事件的时间、位置、震级、深度，不是WDI宏观同facts换端点。首轮只实际验证一个源，**不绘图/QA、不构建train/dev/blind或加载模型**。固定2026-10-01T00:00:00Z至10-08T00:00:00Z（官方start/end都含边界）、服务minmagnitude4.5、earthquake、time-asc、limit50。它定义最早最多50事件的小样本，不是整周完整全球地震率；若返回恰50，是否截断目录保持UNKNOWN。返回N才是源事件分母，50是上限，不伪造缺失事件来补满。

官方[查询API](https://earthquake.usgs.gov/fdsnws/event/1/)支持上述参数，单query上限20000；本scope只1GET、≤1MiB、120s/2CPU2GiB，正常TLS、不代理/认证/redirect/retry，不请求count/version/detail/application，不用滚动feed，不换日期/阈值/地区重试。body cap/JSON头部先核，原namespace和worker lock独占，未知只能reconcile原request。

官方[GeoJSON模式](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php)明确FeatureCollection与metadata.count/generated/api、event.id、mag/magType/time/updated/net/ids/sources及Point[longitude,latitude,depth]；深度km、经纬度角度以API规范为准。程序以整数UTC epoch-ms核时间落在固定窗且升序；原十进制数用Decimal保存不浮点往返，保负深度，不填null；每一原事件槽均保status/null/errors。事件ID和aliases合并成物理事件组，重复alias组全成员不合格，不当独立样本。Geo格30纬×60经只初始区域key，不是已完成来源/余震簇隔离。

官方[版权FAQ](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted)说明USGS自产数据/信息在美国publicdomain并可自由使用，要求保USGS来源；第三方图像可能另有版权。本scope仅政府目录数值事实，保net/sources/API vintage，不抓第三方照片、地图或logo；不捏造CC0或全球无条件版权结论。原copyrights-and-credits页面web研究Timeout保留，未重播该页；FAQ独立可读。未来派生图标明本地绘制/USGS来源，不冒官方图或安全/风险预测产品。

源可达并经独审后，才另scope冻结实际event IDs/bodySHA及版本、按eventalias+地理区域分组，额外预注册近邻时空簇/区域边界缓冲防同一余震序列跨split；所有同事件/图窗/问题共组。盲组需在QA生成前冻结，不按答案或模型效果选；null/错/缺失全分母，问题只从显示的真实数字确定性重算。异种magType不能冒统一能量/风险比较；图表读数明确字段/尺度/单位与tie规则。此轮train/dev/blind、图、QA数量均0，数据集GO仍false。

未来可接同一框架五VLM原生图文模板/messages-images/collator，原17/A100ref不动；首次源成功不是native可读性/监督mask/EOS、训练/恢复或任务效果。原WDI/CORD封存请求/失败及网络限制不变。root审/pub后唯一 `once.py --publication <新pub> --claim-sha256 <CLAIM SHA> --root-go`，结果只忽略local-only，GO全false。
