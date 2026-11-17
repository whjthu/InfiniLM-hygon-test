# USGS v2：仅修复 Job 时钟 IPC，尚未执行

原 request d8705d29037c4a34b6b721e16a1fda86 已知 Job C1，原 traceback 在 gate 比较处失败。原 child 终止、精确 published source 的 gate→lock→result/progress→getter 顺序与 traceback 共同证明 0GET；不以缺文件单独推断。原目录和失败全部保留。新目录独占、新120s epoch，不重放旧 Job，不延原预算。

仅 Job gate 改用标准 JSON float，与 argparse float 同类型精确往返；GeoJSON sample.py 的 Decimal 解析和数值/字段/来源契约字节不变。新 getter仍同一个固定官方 URL、1GET≤1MiB、正常TLS、无redirect/retry/proxy/新日期。2CPU/2GiB Job机制原样复用，未重新执行Job资格或旧科学测试。初次取样不是数据集/训练/效果资格；所有GO默认false。

前置核原4小证据、原 published once.py SHA与控制流、已发布失败blob；未填public_failure则拒绝运行。root审/pub后唯一 once.py --publication <v2pub> --claim-sha256 <newclaimSHA> --root-go。只有root执行。本包准备/测试无HTTP、GPU、模型、Slurm。任意实际GET失败或未知保持封存，不自动重播。

官方API/字段/权利沿用：https://earthquake.usgs.gov/fdsnws/event/1/ 、https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php 、https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted 。仅USGS数值事实；保来源/vintage，不取第三方媒体、不冒CC0。最多50是请求上限，实际返回N是分母，满50不称整窗完整。未来图/QA/切分/native/模型资格均另审。
