# WDI v2：只取三个从未请求的事实分割

v1 原request `22ae4128d8294182834782278956d36b` 已known terminal FAIL，前三份指标元数据完成、第四countries请求TimeoutError。原4次attempt/3完成/0事实/0图保留；v2不是重试countries/元数据，也不重置原epoch。原GET.enter必须精确为 indicator_0/1/2/countries、无任何facts，原intent/outcome/worker/progress及3metadata全SHA核后，只将2434B元数据copy到新namespace；绝不修改v1。

唯一新网络scope为原PLAN已固定而尚未调用的 facts_train/dev/blind，3 GET、总body cap6,291,456B，正常TLS官方api.worldbank.org、不重定向/重试/替代源；socket timeout=min(75s,原epoch余量)，新单epoch300s含原件复核/copy/GET/绘图，WindowsJob仍2CPU/2GiB进程及Job commit硬限（RSS另观测）。原4+新至多3共7次网络分母，不因为去掉countries而隐藏旧失败。任何facts失败或超时即封存路线，**没有第三次数据下载路线**，不缩图/问题/年限来争取完成。

不再请求countries endpoint。24国代码/拆分不变；国家名称从每个官方事实行的country.value提取，用countryiso3code分组，要求非空字符串、同代码全部21行一致、24国完整、各拆分168行。名称不是从代码手工翻译，不补收入或region字段；地理分层仍人工预注册分层，不冒官方分类。其他source/date/units/lastupdated、Decimal显示/QA/tie/缺值、72图/648槽与native262144像素完全不变。图仍本地派生、GDPdeflator不是CPI。

原合格overlay与3.10.7/numpy2.3.5/Pillow13.0.0不重装，字体SHA必须3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22。实际执行用本包 `with_qualified_overlay.py -- --publication <root新pub> --claim-sha256 <最终SHA> --root-go`，它先核原qualified fsid-fix-v2 outcome13615B/SHA d4642ac0... passed/C0，核原published overlay_env.py712B/SHA0daf99cb...，只调用其environment()给本包execute_once子进程；不改shared os.environ，不调用硬编码v1的旧with_overlay，不改原qualified源码。publication必须包含原公开失败摘要精确3827B/SHA17ffc175...，原件仍由worker实际复核。

源码准备、不GET/渲染/安装/Model/GPU/Slurm/Git。root/CPU_data_build/Dataset/training/GPU GO均false；source/pub/admission后根唯一执行。原root-reader/adapter冻结不改，v2另独立reader核cachedmeta+3facts与国家名来源，blind PNG/gold不打开。首次新构建成功也仅数据产物，不是可读性/native/数值/恢复/领域效果资格。
