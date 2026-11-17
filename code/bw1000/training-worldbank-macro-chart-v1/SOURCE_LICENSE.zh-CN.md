# World Bank 宏观图表候选 v1

查证日期：2026-11-17。只查发布方文档、指标元数据和许可页面；尚未获取计划中的504个年度事实，没有新训练或评分。

数据来源为 World Bank World Development Indicators（API source=2）。[WDI 目录](https://datacatalog.worldbank.org/infrastructure-data/search/dataset/0037712/world-development-indicators)与下列三个指标自己的页面均标识 CC BY 4.0，而不是仅依赖网站通用版权声明：

- [GDP growth](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG)：实际 GDP 年增长，年百分比。
- [GDP deflator inflation](https://data.worldbank.org/indicator/NY.GDP.DEFL.KD.ZG)：GDP 平减指数年增长，年百分比，不是 CPI。
- [Gross capital formation](https://data.worldbank.org/indicator/NE.GDI.TOTL.ZS)：资本形成总额占 GDP 的百分比。

[许可汇总](https://data.worldbank.org/summary-terms-of-use)要求保留 World Bank、原数据提供方和适用附加条款的归属；不能暗示认可；第三方例外需要按元数据处理。汇总指向的旧 dataset agreement URL 当前重定向到通用网站条款，不能把该重定向当作独立新增授权。本候选依据上述个别指标许可、WDI 目录与[发布方公开许可说明](https://datacatalog.worldbank.org/public-licenses)；派生 credits 保留汇总条款 URL、CC BY 4.0 URL、实际 sourceOrganization、访问时间/快照和修改说明。源元数据新出现 license/rights/restriction/copyright 字段将硬拒绝，交回来源复核，不自动忽略或改变许可。

原提供方包括各国官方统计机构、中央银行、OECD National Accounts 和 World Bank staff estimates，以每次实际 API metadata 为准；不把所有数据声称为 World Bank 原创。生成图片是对官方数字本地绘制、两位小数标注后的新图表，不是官网截图或原始图像；不使用官方标志。派生问答和标签由构建器确定性计算，不冒充发布方人工标注。

[API 基本调用文档](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581)说明国家/指标多选、source/date/per_page 等参数；[V2 文档](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392)说明公开访问不需要密钥。当前实际读到的指标元数据 `unit` 为 **空字符串**，可见单位来自上述指标定义，不伪造 API 字段。构建器核实际名称/sourceNote/sourceOrganization，空单位与已知模式不符则拒绝。API 是会修订的数据库：截止2020仅限制观测年，不表示2020发布版本；真实 source lastupdated、访问 epoch、7响应原bytes及SHA才构成未来可复现快照。

绘图环境固定 Matplotlib 3.10.7、numpy 2.3.5、Pillow 13.0.0；[Matplotlib 官方版本文档](https://matplotlib.org/3.10.7/)与[维护者 PyPI 发布记录](https://pypi.org/project/matplotlib/3.10.7/)确认所选版本存在。准备时本机 Matplotlib 缺失；现已在独占 overlay 完成真实安装、字体与渲染核验（见 results/bw1000/2026-11-17/workspace-matplotlib-fsid-fix-v2-original-install-and-renderer-pass.json），不改共享环境、不安装全局依赖、不手绘替代。首个 GET 前会检查三个精确版本与原配 DejaVu 字体并记录字体SHA；根协调者必须先选定并审阅一个独立、具备相应依赖的解释器环境。依赖安装不是本次下载scope的一部分，若需要安装应另行冻结 wheel/dependencies/来源与体积预算。
