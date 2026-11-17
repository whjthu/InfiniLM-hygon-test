# CORD Viewer 合法衍生图：只准备一个train槽

新scope允许已公开CC BY4的CORD材料派生转码；保作者/NAVER CLOVA、原数据与许可链接和转码修改说明，不暗示认可，不恢复发布者删去的store_info/payment_info/etc字段。依据[发布者许可](https://github.com/clovaai/cord/blob/327310ce58c1623255821d062b3a759ff3789e3c/LICENSE-CC-BY)和[CC BY4的改编条件](https://creativecommons.org/licenses/by/4.0/)。这不是放宽正式17原配方或重试原metadata API。

仍按原seed/revision/ordinal的哈希规则，先验唯一选择train ordinal795。仅两步：

1. 官方Viewer `/rows`精确请求1行，JSON≤1MiB；X-Revision必须唯一且等于7f0115...，800train/row795/完整ground_truth/gt_parse与truncated_cells/partial全部核。缺失或错revision/格式/截断即失败、0media，不改config/row/URL重试。
2. 只允许这条row直接给出的官方datasets-server资产路径，版本/default/train/795/image/image.jpg或png都精确绑定；至多1张≤8MiB。公共服务返回的expiring locator query可原样使用，仅预列Expires/Signature/Key-Pair-Id，不能刷新/重签/构造；无用户token/key/Authorization/Cookie/登录。URL/query只私有忽略目录，public仅locatorSHA。无redirect/proxy/fallback。

实际Pillow尺寸必须等于rowimage.width/height；若原ground_truth.meta.image_size存在也必须完全相同；维度变化在media前拒绝，不允许Viewer缩图/裁剪替代。完整解码后记录模式、编码、原bodySHA及无裁剪RGBA8规范像素SHA，不本地重存JPEG或改图；原gt_parse和完整row响应只私有。无原图可比较时**没有量化转码损伤**；公开服务源码说明embedded image可能JPEG/PNG重编码，这里明确接受该衍生副本，不声称原body/原像素等价。

未来A100/Hygon与base/domain必须共享本次同一个固化body/像素、标签、数据顺序和评分器，建立新domain recipe配对效果；不能称官方CORD100复现，不能继承A100正式17或旧Treasury/CORD来源资格。初次1row只通路，不是32/16/16数据集、可读性、native mask/EOS、数值/恢复/训练资格。

120秒单epoch包含source检查、2GET和解码；actualWindowsJob2CPU/2GiB commit、Pillow13，无环境安装/GPU/Slurm/Torch/model。原metadata request5a9ee32c108548d89eecb77f324050c5 WinError10060失败保留，不重原URL；Viewer页面TooBigContentError也作为风险保留，不拿来假报本次真实终态。根审阅普通pub后唯一 `once.py --publication <新pub> --claim-sha256 <CLAIM SHA> --root-go`；exclusive原namespace禁止重播。

Primary：官方[rows参数](https://huggingface.co/docs/dataset-viewer/rows)、[cache X-Revision实现](https://github.com/huggingface/dataset-viewer/blob/main/libs/libapi/src/libapi/utils.py)、[转码分支](https://github.com/huggingface/dataset-viewer/blob/main/libs/libcommon/src/libcommon/viewer_utils/features.py)、[image.save](https://github.com/huggingface/dataset-viewer/blob/main/libs/libcommon/src/libcommon/viewer_utils/asset.py)、[公开locator版本路径与签名](https://github.com/huggingface/dataset-viewer/blob/main/libs/libcommon/src/libcommon/url_preparator.py)。这里描述所审公开实现，不假称已核生产部署commit。

本包未调用任何row/asset GET，全部GO false，无新训练loop/大下载器。
