# CORD-v2：官方固定版本文件元数据，一次请求（未执行）

只准备官方HF metadata API的一次新GET：`https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/revision/7f0115a4b758a71d6473b8d085751692da2fef98?blobs=true`。官方Hub API文档支持dataset_info(revision,files_metadata=True)，官方huggingface_hub v2.4.0源码明确映射revision路径及blobs参数，并列出siblings的size/blobId/lfs字段。这里直接stdlib GET，**不import SDK、不发现token、不认证、不代理、不重定向/重试**。

该exact API URL和固定revision tree URL本轮均未请求或失败。`WEB_ATTEMPTS.json`列已知官方web研究URLs；失败的fixed-revision raw dataset_infos URL及失败的main parquet blob页面被去重禁用，不改query/host/文件名绕过。web工具“不可达”只说明工具结果，HTTP状态和origin是否发出不能确认；local direct HTTP GET为0，不把两者混同。新scope请求repo文件身份，不请求旧失败raw metadata内容。

预算只1URL/1GET（小于用户max2）、总body≤2MiB，单epoch120秒含source/pub检查和Job，actual2CPU/2GiB进程及Job commit限额、RSS另采。普通TLS；JSON200、Content-Length/编码/分页在body前拒绝异常；未知长度读到cap仍无EOF则失败，不再探测超限字节。实际namespace exclusive mkdir，UNKNOWN或失败不能重放。错误保metadata1请求/6文件分母和进度，不打开第二URL当fallback。

目标只确认6个具体Parquet的固定revision/path/size/Git blob/LFS SHA与总文件尺寸。publisher LFS SHA是源声明，**不是已下载并hash文件正文**。未发HEAD/Range、未取footer/rowgroup/图片/标注，Range可用性和有界取样成本始终UNKNOWN。未来若需footer/selected rowgroups，必须另外认领发布，并先取得正常TLS/精确206/Content-Range/字节成本证据；不能因为元数据成功就下载数百MB rowgroup或整2.31GB，不用第三方镜像。

候选仍接统一框架native qwen3_vl/messages-images数据接口，之后其他四VLM分别核原生HF provider；此包没有数据adapter/训练loop，不改变正式17或任一sealed/pending路线。32/16/16只是上轮候选计划，没有实际图片、开发分数或领域checkpoint。

根审阅、普通pub及明确root-go之后才可唯一调用：`execute_once.py --publication <本包pub> --claim-sha256 <CLAIM SHA> --root-go`。原JSON与结果仅忽略目录 `local-only/training-CORD-v2-official-metadata-once-v1-original`；此轮执行次数0、全部GO false。

Primary依据：[HF API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.dataset_info)、[官方客户端源码](https://raw.githubusercontent.com/huggingface/huggingface_hub/v2.4.0/src/huggingface_hub/hf_api.py)、[官方固定revision](https://huggingface.co/datasets/naver-clova-ix/cord-v2/commit/7f0115a4b758a71d6473b8d085751692da2fef98)。无GitHub API。
