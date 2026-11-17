# 固定16图公开oEmbed作者/署名来源：准备态

官方文档核验时间及primary URLs见SOURCE_PROOF。Flickr正式getInfo要求API key，本包不用；官方Staff确认oEmbed无需key，opt-out可能仅返回link。原JSON-LD能力未获文档支持，已放弃该假设。oEmbed标准有作者/题名的可选字段，未保证license扩展；不声称16图现在能取得完整署名。

固定原roster16 photoID，只用官方Flickr base58字母表纯计算 https://flic.kr/p/<encoded>，作为 https://www.flickr.com/services/oembed/?format=json&url=<urlencoded> 参数。provider registry一手列出的输入scheme与endpoint支持此组合。不请求短链/照片页面/图片，不读密钥/登录cookie，不发REST请求；每槽一GET、正常TLS、exact200/endpoint、不redirect/fallback/retry/backfill，最多4并发，每JSON64KiB/总1MiB。未知长度恰好达到64KiB时拒绝，不加读一个字节超预算。非JSON、optout/link、作者/题名/同photoID/canonical关联缺失或不符保失败16分母，不换图/登录/绕过。原qualified Job2CPU/2GiB commit limit、新600s单epoch、deadline-5清理不增。

真实author_name/author_url/title单独资格。响应图URL必须同固定photoID，canonical web_page若存在须同owner/photoID；没有时按已文档化规则从实际author owner和photoID派生并明确标derived未GET。license_url不是标准保证：缺失/不同CC许可始终pending；仅实际同CC BY2 URL可作当前license来源证据，配合同照片作者/题名才credits sourceproof。COCOlicense4或数字app/license字段不补缺、不猜映射。producer passed仅表示16作者/题名均取得，不表示16license齐全；两种分母分别报告。

全raw JSON/作者/题名/URL/ID只留忽略local-only；JSON的HTML/图片URL不执行或拉取。原准备仅源码/AST；独审新增门禁获授权执行一次纯stdlib合成10正常/10拒绝检查，旧测试不重跑，不取16photo/oEmbed、不重读旧partial/原references/图片。原generic caption dev16不是领域训练集，不更改正式17项、原训练/开发或timeseries53 pending。native/scorer未运行，即使credits齐全仍Dataset_GO/training_GO/GPU_GO=false。

根发布后唯一命令：

```powershell
& 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/code/bw1000/training-treasury-caption-dev16-oembed-credits-v1/execute_credits_once.py' --publication <PUB> --claim-sha256 <CLAIM>
```

独占local-only/training-treasury-caption-dev16-oembed-credits-v1-original；existing拒绝。actual gitshow原metadata成功966516/新pub sameblob与ownMF/claim/clean和原Job字节、冻结roster+真实媒体独审SHA均在HTTP前。若缺实际公开许可字段，保持pending并停止此槽，不追加另一条网络路线。

新增门禁：JSON parse_float 额外拒绝 1e999 等溢出非有限数，duplicate key 与 NaN/Infinity 拒绝不变。许可只接受实际明确 CC BY2 URL；出现额外 license/licence/licence_url/license_name/license_type/rights/copyright 声明时，除同一认可 URL 外均视为未解释或冲突，作者/标题可保留，许可仍 pending，不做数字或文本映射。独立 reader 对保存的 MIME/Content-Length/Transfer-Encoding/Content-Encoding 与实际完整 body 长度重新检查，明确不是独立实时 HTTP 观测。
