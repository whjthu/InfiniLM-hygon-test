# 固定16图媒体 staging：准备态

仅从已独审冻结的16图 roster 按原顺序读取，不重选或补位；新namespace独占，原已结束caption请求、partial、reference与训练原件均不读取或修改。相同原COCO bucket的固定AWS path-style TLS URL，分别GET /images.cocodataset.org/train2017/{fixedID}.jpg；不向原HTTP/CDN地址请求，不HEAD、不redirect、不proxy/fallback/retry。每图只一次GET、必须200/single Content-Length≤2MiB/image/jpeg/identity/single actualETag，读body前过headers。最多4并发，合计最多16请求/32MiB，新600秒原epoch贯穿、同已实证Job2CPU/2GiB、deadline-5终止清理，不承诺网络速率。

真实媒体按完整实际字节SHA和Pillow13被动JPEG header/原尺寸核。JPEG SOI/EOI必须存在；不做像素解码、裁剪或resize。16内相同SHA的所有成员拒绝，旧64实际SHA重合拒绝；近重复检测未执行。每图异常/缺失保全16分母，原privatepartial和每图进度留底，失败不续取/补图/重试。controller失败按合格0/16保守记录；worker保存实际完整尝试记录/原始成功数量，不洗失败。

来源/credits：private credits保存官方COCO URL、实际TLS URL、license4/CC BY2链接、Flickr静态来源与photoID。原metadata没有author/title，明确UNKNOWN、attribution false；不根据静态URL猜作者、不把license4或网页链接当完成署名。最短后续是在独立有界scope读取每个固定Flickr photo.gne?id原页面，核真实作者/题名/许可证并保存身份和证据；本包不发该请求。作者证据缺失/不可访问/许可证不符仍未资格。32MiB原jpg、caption、credits和ID清单只留忽略local-only，源码不含原ID或caption。

本轮仅新engine CPU门禁：合成2×3 JPEG、fake HTTP，4正常/20拒绝，真实HTTP和真实图片读取0。原Job源码逐bytes复用其已发布/实测版本，未再执行Job或大内存probe。Pillow版本及四精确provider文件pins在CLAIM，launcher实际前置核。正式原17项验收不变，不触碰timeseries53 pending、模型/processor/Torch/scorer；即使16图全部stage通过，rights/native/CIDEr仍PENDING、Dataset_GO/training_GO/GPU_GO=false。

根发布后仅一次：

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-treasury-caption-dev16-media-stage-v1/execute_media_once.py' --publication <PUBLICATION> --claim-sha256 <CLAIM_SHA256>
```

launcher actual gitshow原成功966516与新publication同一successblob、自己MF/claim/clean、原Job源，然后精确roster/旧64身份及Pillow provider后启动一次worker。输出local-only/training-treasury-caption-dev16-media-stage-v1-original，已存在即拒绝，UNKNOWN不重播。原成功数据资格仅metadata，并非16图已下载。
