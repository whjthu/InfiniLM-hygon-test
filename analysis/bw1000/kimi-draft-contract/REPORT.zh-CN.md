现有 DFlash2 应继续，尚无证据应因 draft 契约而更换权重或重新加载目标模型。当前离线四段结果不是服务接受率，不能证明目标精度、草稿质量或性能上限。

已证实的契约：

- 本地已保存的 checkpoint revision `e77935fb4804e17eb55085bffd045eae1d779769`，config SHA256 `aac0900105b9099b99a452c728ff9c4685889d264f7bed0101e57788df8f09c8`。5 层，target taps `[19,37,66,78,90]`，H7168，`fc_norm=false`，MLA+4SWA/1Full，block8，7 个 proposal，mask163592。作者模型卡指定目标 `moonshotai/Kimi-K3`。[固定模型卡](https://huggingface.co/lightseekorg/kimi-k3-dflash2/blob/e77935fb4804e17eb55085bffd045eae1d779769/README.md)
- 原作者训练实现的上下文只允许 anchor 位置之前的特征；query positions 为 anchor+[0..7]，slot0是已知 anchor，其余七行预测同位置 token。故当前 scorer 的 features[0,N)、anchor at N、hidden/logits[:,1:] 与公开实现一致，不应套用 DSpark 的 query 数规则。[训练掩码及标签](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/models/dflash.py)
- selector 训练使用行1..7、真前驱/后继；推理用已知 anchor 起步，逐步选出的前驱更新 Markov 路径；不能用所有行独立 raw argmax 替代。当前 scorer 有该顺序，且没有向推理传入 training_successor_ids。[selector 训练](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/models/dflash2.py)

仍然 UNKNOWN：精确训练目标 revision、目标量化格式、历史训练运行的 feature stream。完整 embedding 字节一致仅绑定 embedding，不能绑定全部专家/注意力/AttnRes 权重。当前目标本地 metadata config SHA `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` 声明 compressed-tensors MXFP4 pack/group32、AttnRes block12，不能据此反推训练目标。

AttnRes 证据有范围差异，不能将某篇文档直接当作此 checkpoint 的生成记录：

- 作者 [AttnRes 捕获说明](https://github.com/lightseekorg/TorchSpec/blob/main/docs/kimi_k3_attn_res_capture.md) 描述下一层 consumer score 定义的 mixture，输入 norm 前，YAML id 会加1。它不是普通 hidden_states 通用索引能表达清楚的值。
- 作者当前 [TokenSpeed 模型说明](https://github.com/lightseekorg/tokenspeed/blob/main/docs/recipes/models.md) 描述默认 completed-layer prefix；明确声明 `aux_hidden_stream=attn_res` 的 draft 则选 mixture。当前 pinned checkpoint 未声明此字段。这支持当前实现选择的兼容性假设，但未证明历史训练运行。
- 发布文链接的 [训练 recipe](https://github.com/lightseekorg/TorchSpec/blob/main/configs/vllm_kimi_k3_dflash2_mla_swa_gb300.yaml) 目标路径是占位符，且 draft 为6层6taps（包含54），与现有5层5taps不同。其24K配置也不能替代发布文64K概述而宣称精确历史训练长度。[recipe draft config](https://github.com/lightseekorg/TorchSpec/blob/main/configs/draft_models/kimi_k3_dflash2_mla_swa.json)

建议只执行一个选择：保留现有7.386GB权重，继续当前 verifier 正确性定位；同一次必要的目标运行中，若真实模型已有 AttnRes bank/consumer mixture 可低成本导出，额外保留这五层的 prefix 和明确标记 consumer 的 pre-norm mixture 两套特征，再在同一已驻留 draft 上比较。不要为 UNKNOWN 单独重载目标。当前四段 prefix 离线结果5/1/1/2只说明这一真实轨迹上的匹配，不应用来挑选最有利契约或推断总体接受率。未来用事先冻结的多提示集对比，最终指标仍为正确生成下的实际 decode 用户 tok/s。

成本与收益判据：已验证完整草稿远端加载约20.43秒；当前完整重算约19.5–20.2ms。新增远端权重准备0字节。若平均接受 a 个 proposal，则一轮最多输出 a+1 个 token；只有 `(a+1)/(draft+verify+capture+replay/communication seconds)` 超过普通 AR，才有用户速度收益。该式不允许用公开接受率直接预测本机性能。现 checkpoint span8不能通过配置改成已训练span16；本次未找到可证实的同作者同 checkpoint span16训练产物，不扩展模型搜索。

源码网页 main 会变化，精确历史 commit 仍 UNKNOWN。本目录保存实际读取的小型源码及逐文件 SHA，详见 source-manifest.json；无模型权重、无原始激活、无 SSH/GPU/Git mutation。
