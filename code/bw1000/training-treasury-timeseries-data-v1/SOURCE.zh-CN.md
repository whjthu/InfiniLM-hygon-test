# 财政时间序列表格与图表数据准备

数据由美国财政部 Bureau of the Fiscal Service 的 FiscalData 平台发布，采用 Debt to the Penny 官方API的日度债务数据。官网 Licensing 明确允许复制、改编和再分发，涵盖商业及非商业用途。本包使用原数字自行渲染表格和带精确标注值的折线图，保留发布方、来源和改编说明。图像属于本地派生插图。[FiscalData Licensing](https://fiscaldata.treasury.gov/about-us/#licensing) · [官方数据字典与限制](https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/debt-to-the-penny)

三个金额字段分别为debt_held_public_amt、intragov_hold_amt和tot_pub_debt_out_amt，源metadata给出CURRENCY及$10.20格式，按原始名义美元、两位美分使用。原数据字典说明总额等于公众持有债务加政府内部持有额，且JSON保留美分。页面提及的“百万美元”对应其他报表的说明，不能用于缩放此源。本包不做币值单位或通胀调整。[官方API文档](https://fiscaldata.treasury.gov/api-documentation/) · [Open Data Policy](https://fiscaldata.treasury.gov/data/about-us/901-1%20Open%20Data%20Policy.pdf)

实际复用现有173,961字节官方JSON缓存，SHA256为fba621dcbd93771067c0b8c8448ce6032fef5f5b44dc8ca4bc4198033d72c2fe。完整缓存包含1,125个有序日期，本次新包只派生120个原日期行，保存的selected-source-rows.json为18,583字节，未再次下载数据。历史缓存中的2025年在原数据构建器中被排除，旧160训练与108开发输入保持各自身份。固定截止2025-12-31，按record_date升序，分别取2025Q1、Q2、Q4的首40日期给train/dev/blind。每个分割使用8个不重叠的5日期窗口。同窗的3字段×3问题类型×2媒体形式共18题，全部绑定同一个窗口和分割。复现查询的fields、filter、sort、page[size]=40及分页参数由queries()固定。它们是缓存子集的可复现选择参数，本轮未声称重新请求了这些API响应。

每个分割为40日期、8窗口、16图像、144题，其中lookup、signed_difference、maximum_date各48题。共120日期、24窗口、48图像和432题。数值按Decimal精确重算，最大值若同值取最早日期。图表分面各自缩放，逐点给出完整原USD数值。每道题声明只输出两位小数金额或日期，开发和盲测的inputs.jsonl只含用户问题，金标独立存labels.jsonl。自动CPU检查校验全部派生答案和媒体身份，未进行盲标人工分析、训练或调参。

私有manifest、原始行、问题、labels和PNG全部位于local-only/training-treasury-timeseries-data-v1。Git适合维护本builder、CPUcontract、来源说明及不含逐题内容的DATA_QUALIFICATION.json。原始manifest包含逐图哈希与源窗口身份，应留在数据域。图像由Pillow按冻结字体渲染，字体仅记录SHA，没有分发字体文件。验证了跨split的日期、窗口、图像身份交集为0。窗口同源图表变体严格保持分组，这个24窗口小样本尚不足以外推财政领域效果。

该包只完成数据资格。尚缺原生Qwen processor实际输出、有效监督token与图像缩放后的可读性、模型数值/训练/恢复/性能及领域效果验证。将来若接入训练，应使用统一框架的原生Qwen插件和collator建立独立数据配方，保留与现有160/108输入分开的对照身份。本轮不会生成第三模型candidate，不改既有328评测流程。

领域KPI应分列数据合格、实际训练与恢复、正式同配方性能和隔离开发/盲测效果，准备包不计训练完成或模型产出。
