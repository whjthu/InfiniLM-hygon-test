# Qwen3-VL 财政表格问答试验

数据源为美国财政部财政服务局发布的 *Debt to the Penny*。仅使用日期和三列财政金额，财政部提供公开机器可读数据，Data.gov 条目标为 public。其美国政府官方作品依 17 USC §105(a) 在美国不具版权保护。表格图像、题目模板和答案由本工具生成，无第三方图片或问答标注。正式对外发布另核适用地区和模型权重条款。[财政部数据字典](https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/debt-to-the-penny) · [Data.gov 登记](https://catalog.data.gov/dataset/debt-to-the-penny) · [财政服务局开放数据政策](https://fiscaldata.treasury.gov/data/about-us/901-1%20Open%20Data%20Policy.pdf) · [17 USC §105](https://www.govinfo.gov/content/pkg/USCODE-2024-title17/html/USCODE-2024-title17-chap1-sec105.htm)

2026-11-17 已获取 [1,125 条历史行的官方 JSON 查询](https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?fields=record_date%2Cdebt_held_public_amt%2Cintragov_hold_amt%2Ctot_pub_debt_out_amt&filter=record_date%3Agte%3A2022-01-01%2Crecord_date%3Alte%3A2026-06-30&sort=record_date&page%5Bsize%5D=2000)。API 可能修订历史记录，以单页完整性和归档内容 SHA-256 `fba621dcbd93771067c0b8c8448ce6032fef5f5b44dc8ca4bc4198033d72c2fe` 绑定本次试验。源码的 `--source` 接收该本地归档，`--font` 接收明确的字体文件。首次本地构建使用 Pillow 13.0.0，Consolas 字体 SHA-256 `c6e6ce8119fdd47ec6a5449a08e2d2ad7f41ea03143aae193068ed9fa58eaebc`。字体原件只用于渲染。`--output` 必须为全新的目录。原始 JSON、图片、问答和盲评标签留在忽略的 `local-only/`。

| 划分 | 日期 | 表格 | 问答 | 每题型 |
|---|---|---:|---:|---:|
| 训练 | 2022–2023 | 72 | 216 | 72 |
| 开发 | 2024 | 36 | 108 | 36 |
| 项目盲评 | 2026 上半年 | 18 | 54 | 18 |

每月从不相交的五行区块固定选择三张表。按日期查值、计算有符号差值、寻找最大值日期三类题均衡，答案通过 `Decimal` 与确定的并列规则生成。2025 年预先留空的 250 条中有 1 条三字段恒等式异常，未进入样本。开发和盲评输入为用户消息和图片，答案保存在独立评分文件。盲评约束由项目流程执行，财政部原数据公开。固定版式与单一来源限制外推，后续可加独立来源和不同版式。

本机 CPU 独立验证成功解码并重算 126 张 PNG 的 SHA-256，核对了 378 条输入、文件摘要、每个划分内的题型配额、图像身份和训练/评估答案通道。汇总清单 SHA-256 `8a888584e40c7b2bbfea54824a2e80783b05bd166d759133863b435285e1ab4e`。

## 下一次验收

新增领域配方复用统一训练框架的 Qwen3-VL 原生模型、图片 processor 和 `qwen3_vl` collator。原 T01 配方、数据和既定启动 SHA 保持冻结。海光 CPU 步骤 `1103606.152` 已验证 160 条输入的原生处理、视觉与标签掩码，以及不等长双样本的批处理。设备试验采用原 T01 单节点八卡 BF16、每卡 batch 1、累积 4、视觉骨干冻结的设置，首批 5 步共需 160 个样本，从 216 条中固定选取。设备首跑前固定 LoRA 初始化、参数范围和版本，随后记录实际更新、峰值显存、最慢 rank 计时和 step 2 新进程恢复。当前暂无设备训练、恢复或效果证据。

`prepare_qwen_inputs.py` 接收已冻结的本地数据根 `--source` 和容器数据根 `--target-root`，以独占创建方式写 `--output`。它核对训练清单、输入、答案及图片哈希，按稳定哈希顺序选取 160 条，题型配额依次为 54、53、53。输出给 Swift 的图片路径只指向目标根下的 `train`，开发与盲评标签保持隔离。实际私有训练 JSONL 的 SHA-256 为 `e0219347307a94259b826b929a88f3fe991b688026677c86c25075c6941bc19b`，以当前预定挂载根为准。选样顺序已冻结，原生分布式 sampler 的实际 rank 与微批顺序仍须记录。

开发集完成算法选择后再开放盲评评分。对照同一初始底座和相同更新预算的通用数据 LoRA，报告三题型及总体准确率，按表格做配对重采样，并报告无图对照以区分表格理解与文本猜测。金额严格比较数值和符号，允许等价的千分位格式。日期按 ISO 字符串比较。缺失、格式错误与错误答案进总分母。通用能力继续用既有 TextVQA 和 COCO CIDEr 评分器记录变化。领域增益报告置信区间，保留失败例和可追溯 checkpoint/adapter。设备速率和实际占用另记，领域配方独立于原 17 项验收结果。
