本包是新的 USGS 多事件输入配方，尚未抓取、渲染、编码或训练。旧 150 次预测及已暴露标签不参与选择；旧训练、开发、盲组不改名或合并。

2026-11-19 只读核验的发布方来源：

- https://earthquake.usgs.gov/fdsnws/event/1/ ：历史查询支持明确 UTC 窗口、贡献者、reviewed 状态、time-asc 排序和 limit；查询上限为 20000，本次自行固定 1000。返回达到 1000 即视为可能截断而停止。
- https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php ：字段包括事件 ID、别名、mag、magType、time、updated，以及经度、纬度、深度；深度单位 km，时间为 epoch 毫秒。query 的 metadata.count 可缺失，不能用 limit 冒充实际返回数。
- https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits ：USGS 制作的数据属美国公有领域，但网站内第三方照片等可能例外。本包只取 US 贡献且 reviewed 的数值记录，不取照片、外部产品或标志。应署名 U.S. Geological Survey；将来的图与标签由本地程序派生，不能称 USGS 原图。

固定 2026 年 6 月（截至 30 日 23:59:59.999Z）查询及预算见 PLAN.json：一次正常 TLS GET，最多 4 MiB；不请求 count、详情、分页、重试、代理或替代窗口。最多 999 条实际事件不是计划实测数量。API 的首选事件值可以修订；需保存完整响应 SHA、generated、updated、实际获取时刻和 API 版本。本数据不提供风险、能量或伤亡判断，异 magType 只比较标明的报告数字。

整理入口为 prepare.py。先联合旧、新事件元数据，按共享别名及 72 小时/500 km 邻近关系传递闭包，排除触及旧事件的组件；然后对新数据追加 30°×60° 地理格闭包，先按固定哈希分 train/dev/blind，再分连续、不重叠六事件窗口。旧源只读事件字段，不接收旧问题、标签、预测。

每窗口必须有 4–6 个事件，显示到两位小数的 magnitude 与 depth 都非恒定。这是事先登记的输入多样性条件，不以模型表现或答案是否容易挑选。短尾和常量窗口记录拒绝，不插补或换查询。计划 12/6/6 图，每个 dev/blind 至少两个独立组件；不足就失败并保留缺额，不训练。

此处明确采用新空间—时间规则，区别于旧数据跨月份永久排除相同 coarse-cell 的规则。同地跨月仍可能相关；不能称全球新地域 OOD 或绝对无泄漏。贡献者/震级/时间截断亦限制代表性。正式 17 用例配方不受本包改变。

后续 QA 合同预先固定为每图六类：指定事件深度查找、首尾显示深度差、最大深度全部并列、最大报告震级数字全部并列、深度至少 100 km 的事件数量、最早事件标签；阈值问题可自然得到零，不据此重选窗口。gold 全由实际显示 Decimal 数值计算，ROUND_HALF_UP 两位；并列按表顺序完整列出。新评分仍要求严格类型、自然 EOS、cap/missing/wrong 全分母，不能以 loss 或有效解析代替任务正确率。

图片必须之后独立冻结尺寸、字体、renderer/body SHA，并做人工可读性及原生完整输入资格；当前没有生成图片、gold、模型检查点或新的效果结论。未来输入继续使用统一框架 messages/images，训练仅 train 的 assistant 监督，dev/blind 只 user，禁止将 gold 送推理；自然 EOS 由原模板负责，不手添特殊 token。性能须新配方同口径参考，实际 token 曝光另报。

执行缺项：PLAN 三份旧 source 元数据（50/200/wide500）已有 root 提供的精确 pin，重叠快照去重、不同快照保留别名闭包见证。现已接入原 Windows Job 精确副本与一次 HTTPS 实现；本包尚未执行。所有准备、GET、关闭、解析和 fsync 都计同一 90 秒预算、2 CPU/2 GiB，Job deadline 为原 epoch+85，预留最终写盘；没有另起 epoch。原始 JSON、组清单和未来媒体/gold 均限 ignored local-only/R，Git 只源码、小身份与统计。

退出：只允许一次冻结查询；失败/容量/截断/不足均封存，不改窗口补齐。本阶段仅做数据资格。若后续固定三检查点的独立新盲组仍无语义配对优势，不通过重复小盲组或训练步数扫描制造晋级。
