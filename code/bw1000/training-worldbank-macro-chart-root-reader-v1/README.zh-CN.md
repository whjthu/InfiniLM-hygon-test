# WDI 独立原件读取器（尚未执行）

`read_original.py` 不 import producer、Matplotlib、Pillow、Torch、网络或模型入口。未来原作业实际终态后，由根给定已知 intent/outcome/worker 的三个SHA和实际publication/sourceMF，独占新私有输出目录执行一次：

`python -B -X utf8 read_original.py --run <原run> --output <新忽略输出目录> --intent-sha <原SHA> --outcome-sha <原SHA> --worker-sha <原SHA或MISSING> --publication <实际pub> --source-manifest-sha <实际scopeMF>`

核单epoch300秒、已知Job终态/2CPU/2GiB commit、原7响应及对应URL/长度/SHA、504/72/648分母、源版本/归属、train+dev432标签与实际输入、Decimal显示值的独立重算、同组成员/缺值、PNG完整文件SHA与被动尺寸/dpi/Software/chunkCRC。40MiB读取/60s预算；无像素解码，不重选全池、不重新绘图或请求数据。失败producer只消费终态与完整计划分母，不伪造产物资格。若读取器本身schema/pin拒绝，保原件，不允许重作业。

blind原JSON只作原始快照SHA，不解析值；blind inputs/gold/PNG仅stat，**不打开、不核其实际内容SHA或语义**。报告明确区分producer声明与独立实核，不能称648标签全部独审或盲评。实际renderer overlay publication/MF必须显式传入，原base PLAN/MF仍按已发布13f225精确核，不能用同名改变数学。

视觉检查固定6槽：train/dev各3主题，每个主题在全部8个原chart ID中取SHA256(`WDI-visual-QA-v1:<split>:<chart_id>`)最小者。选样只依赖冻结ID，不看数值、答案、可读性或模型效果；失败槽保留、不补位。输出选择清单不自动打开图片。根后续仅打开这些最多6张train/dev图，按原512×512/native262144像素检查标题/单位/年份/7点显示值/负号/小数/遮挡/边界/归属是否可辨；任何一项不明标FAIL或UNKNOWN，不能以本读取器被动PNG检查称可读。blind图/gold禁止用于此定标。

本目录仅准备源码+AST核，不消费实际产物，不跑旧tests。Dataset/training/GPU GO false；无效果或通用能力结论。
