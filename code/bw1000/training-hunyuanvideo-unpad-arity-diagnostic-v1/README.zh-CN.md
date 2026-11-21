# T09 原 .763 解包失败：只读源码定位与独立 CPU provider 检查

原GPU .763 FAILED2:0、0/25保持；pre.762/post.764独立清理不能洗训练PASS。原manifest没有traceback，只保存run_backend阶段和异常文字，所以本包不声称恢复了已消失进程的真实栈，更不盲加unsqueeze或改模型/优化器。

已实际读取root提供的四个当前target源码，并逐字节核原38 binding：transformer39361B/d1a70289…、refiner10490B/02675467…、attention12909B/73db640a…、flashhelper3909B/e9835a76…。这些是当前实际/A100包身份，attention与原官方upstream不同，不能只凭upstream当目标原件。原件只留local-only，Git只本包自写工具及小证据。

源码AST确认：text refiner forward第121行调用attention而未指定mode；当前attention默认flash。Transformer.set_attn_mode只设置self.attn_mode/double_blocks/single_blocks，未覆盖该text refiner调用。flash helper第29/65/68/71行要求unpad_input五返回值；Transformer forward第696行也有五轴shape解包。前者是具体可达候选，实际选择仍受maybe_fallback_attn_mode与安装依赖影响；不能凭相同异常文字唯一定位。原backend明确验证latent五轴并构造cond/cat，不据此删维或加维。

`source_probe.py --binding ORIGINAL_BINDING --source-root ROOT_FOUR_FILES --flat-source --output NEW_PRIVATE_JSON` 只AST，不导入Torch/provider。可另接 `--provider-path .../flash_attn/bert_padding.py --provider-sha256 ROOT_ACTUAL_SHA` 分析实际安装函数的显式return arity；未知表达式保UNKNOWN。四个原文件的目标行号来自本次真正原件，不使用web解析后的不同显示行号。

独立target CPU入口为 `probe_provider_CPU.py`，固定新label `training-hunyuanvideo-unpad-provider-CPU-v1`、request `942d670a2941466b93b9e686f66ec5ef`。root发布后可用原qualified CPU容器/同DTK env和PYTHONPATH启动，0GRES/2CPU、外层建议60s/8GiB；程序内部40s alarm，外层仍须真实持有/回收/终态。**不使用-I-S抹掉原native site**。本agent没有实际执行这个Torch CPU入口。

不猜 `/training-runtime/site` 有flash_attn：先通过当前active sys.path的PathFinder定位package和bert_padding子文件，不执行package __init__。只run_path该实际Python文件，hash前后匹配并核函数sourcefile；不加载hyvideo、模型、权重或native完整forward。指定了独立SHA则严格核，不匹配不能换path/依赖；也不安装、下载或fallback。实际供应者未事前独审时清楚标记observed current source而不冒原A100 provider同字节。

真正输入为CPU float32 arange(24) reshape(2,3,4)和mask[[1,1,0],[1,0,0]]。实测tuple数、每项shape/dtype/CPU/finite，并核unpadded值等于原mask选中行、indices=[0,1,3]、cu_seqlens=[0,2,3]、max_seqlen=2；再用同一精确provider内的pad_input逆变换，核2×3×4与原输入mask保留区逐值一致、padding区为0。四字段语义通过与旧五字段解包兼容分别报告；返回4并不代表当前provider自身算错，且仍非原GPU实际栈证明。

三种visibility=-1后才import Torch，无cuda可用性/device_count/SDK探针；只显式CPU tensors。隐式loader GPU use保UNKNOWN，不冒设备使用已独审0。模型/更新均0。

root新RUN需独占创建在 `ROOT/results/training-hunyuanvideo-unpad-provider-CPU-v1`。CLI `--output RUN/provider-result-private.json`，intent-private.json在任何provider尝试前独占落盘，失败不重放。结果JSON独占写盘+fsync后，stdout才发唯一前缀 `T09_UNPAD_PROVIDER_CPU_RESULT_V1:` 及精确结果bytes/SHA；其它import日志不作为JSON解析。若前缀/返回通道丢失，root应消费原落盘文件/原会话，不重做.785或新provider调用。

`check_CPU.py`仅验证同错误的纯Python协议复现、AST返回分支、active PathFinder与错SHA在Torch前拒绝；不称真实provider CPU已通过。当前0新GPU/训练，不修改任何原数学或配方。

官方固定primary源码（只用于出版来源对照；当前target正文另以实际pins为准）：

- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/models/transformers/modules/token_refiner.py
- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/models/transformers/modules/attention.py
- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/utils/flash_attn_no_pad.py
