v2 极窄布局修复：仅上面板(.14,.63,.82,.21)、下面板(.14,.27,.82,.23)。v1 工程/数学PASS与视觉FAIL均保留，不在旧namespace重跑。详见SOURCE_REUSE.json逐bytes证据。未来独立审计只对v1六JSONL做SHA及逐bytes相等比较，不解析其旧gold；新v2 gold数学审计范围不变。当前无新图或实际执行。

本包初始全部 GO=false，尚未生成。仅消费原 v3 实测选窗，原声明11528B/fe985561...和窗口51038B/f0e611c4...逐 SHA 核验；不改变事件、窗口、3/2/2 闭包分割或 seed。不读取旧 gold、预测或模型；HTTP、native、训练均为0。

入口：固定本机 Python 执行 with_qualified_overlay.py -- --publication PUB --claim-sha256 CLAIM_SHA --root-go。wrapper 复用已实证 Matplotlib overlay outcome13615B/d4642ac0...及环境源712B/0daf99cb...，不安装、不修改共享 os.environ。root 必须先普通发布源码，采用 PLAN/CLAIM 的 root/CPU/READ GO，训练、GPU、Model、Dataset 仍 false。当前publication须包含5267B/907f051e...原输入资格结果（已pub809cdfe8...）。

总120秒自 wrapper 首行 origin 起，所有核验、加载 renderer、24张PNG、144QA、fsync及独立审计均不重置时钟。顺序两个 Windows Job（构建及独立audit进程）各两CPU/2GiB commit，deadline分别原epoch+95/+115，整体+120。qualified Jobhelper字节复用，RSS只报实际OS采样口径。独占新 local-only/training-USGS-multievent-render144-v2-original，未知/失败不重播。

PLAN 的 protocol_pins 对 contracts.py、response_protocol.py、frozen_score.py 固定完整字节。任何gold/图片生成前，先核协议并独占写入 protocol-frozen-private.json 后fsync。六kind为显示报告震级数字max/min全部ID集合、显示深度max/min全部ID集合、E06-E01显示深度差、E06-E01显示报告震级数字差。全部数值先两位ROUND_HALF_UP，gold严格从显示字符串的Decimal计算；并列及零变化全部保留，不筛掉容易或零答案。

response_protocol语法沿已冻结wide版，只有六kind登记及reported_number_change禁止物理单位的必要差分；不根据旧输出改解析格式。frozen_score.py精确复用旧严格EOS/cap/全失败分母评分函数。未来同一输入采用完整4096 context、greedy cap256；cap/不自然EOS/missing/parsefail均不得计正确。新六kind及USGS派生数据不冒TextVQA/COCO/原正式17效果。

每图512²=262144像素，100dpi，固定DejaVuSans字体SHA；上下两个panel展示六原事件E01..E06的两位数值、震级类型，含UTC顺序、单位、网络及来源页脚。属于本地派生图，不是USGS官方原图；异magType仅比较报告数字，不解释能量或风险。

产物24PNG、train/dev/blind的72/36/36条user-only inputs及分离labels，精确30个数据文件+manifest。完整messages/images接口、labels=None，无答案输入或手添EOS。raw、gold和图片全在ignored local-only，Git只源码和小身份。任何中途失败都留build-progress与计划24/144，不补图、不换事件。

独立audit.py从原窗口独立重算显示Decimal、四种全并列和两个变化量，核144数学、source/protocol及30文件SHA、image与question绑定、closure/event无交叉。Pillow核24图字节、PNG结构、尺寸、Software和非空，仅证明该范围，不冒语义OCR或人工可读性。少量train/dev人工视觉审核、后续全输入nativeCPU资格均另行必须；无领域模型/训练/效果晋级。
