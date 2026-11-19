这是 v2 真实失败后的新离线输入配方，所有 GO 初始为 false。原 v2 请求、HTTP 成功和 selection FAIL 均保留，不能将 v3 的结果反填为 v2 PASS。

一次执行：固定本机 Python 调用 once.py --publication PUB --claim-sha256 CLAIM_SHA --root-go。root 先普通发布完整源码并采纳 PLAN/CLAIM 中 root_GO、CPU_GO；source_GET_GO、训练、GPU、Model、Dataset 仍为 false。父控制器首行产生唯一 epoch，含 Git/源码核验、缓存读取、Job、解析、选择和 fsync 的总窗口 90 秒；Job 使用原 epoch+85，两逻辑 CPU、2 GiB commit 硬限制，RSS 另为采样实际值。独占 local-only/training-USGS-multievent-offline-v3-original 目录，未知或失败不重播。

仅读取固定 508506B/dfe601519fbb7229edfd2debfce48512da01365eb79b17a58bba67779a6046bd 已完整保存快照，HTTP=0。发布物同时必须含原 v2 失败 4732B/2e0f2673e096fa40b04175ee2bea0caf047835c0b30e9678f5bb19cd7fae318d。三个旧元数据源 pin 不变，输入 750 条；同事件相同快照去重，不同历史快照仍作为别名闭包见证，不读旧 gold/媒体/预测。

v3 对全部 711 槽重新审核；US 贡献者通过 sources 中包含 us 判断，同时保留 preferred_net 与完整 sources。contributor 不等于首选网络 net。旧 8 个拒绝槽单独登记 v3 是否合格；任何实际无效槽保留原因和 711 分母，不以排除后的数量声称全源通过。USGS ComCat 公开数值包含不同网络贡献，不声称所有首选解均为 USGS 原创；保留发布方及网络署名，数据更新与异 magType 限制不变。

闭包保持旧/新别名与 72h/500km 联合排旧，再为新源加 30°×60° 地理闭包。先按不变的 seed/hash 次序生成每组全部合格输入窗口容量：每窗 4–6 个事件，两位显示的震级和深度均非恒定。此时无 QA、答案或模型输出。

分配采用有界 memo DP，窗口容量截断到 12/6/6，每 split 的有窗组计数截断为 2，最多 400000 状态；固定 train/dev/blind 尝试顺序，只求一个可行解。无解或状态/时限耗尽即失败，不换 seed、不绕过分组、不穷举 3^N。每组完整分配到唯一 split 后固定 roundrobin 取窗，实际每个 split 至少两个有窗独立闭包，禁止把无合法窗口组计入覆盖。

独占写盘顺序：source-audit-private.json → input-selection-declaration-private.json 并 fsync/记录 SHA → input-windows-private.json → worker-result.json → outcome.json。declaration 绑定缓存、旧源、seed、组分配及窗口身份。只产出输入窗口；QA/gold/图片/processor/模型均为 0，后续生成必须另 scope 引用此已封 SHA。train/dev/blind 计划 12/6/6 及失败缺额始终保留。成功只意味着输入选样可行，不能晋级数据可读性、训练效果、性能或正式 17 用例。

官方语义依据：https://earthquake.usgs.gov/fdsnws/event/1/ 、https://earthquake.usgs.gov/data/comcat/ 、https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits 。源码说明及本地纯 DP 合约不替代真实 711 槽审计。所有原始记录、窗口和组 ID 留 ignored local-only；Git 仅源码、小身份及计数。
