本版本提供五个不同模型的真实数据/目标训练候选及五VLM共用的数据接口，不新增训练loop，不改变正式17配方。CANDIDATES.json绑定当前本地原件SHA，区分已有数据与“新训练scope真正可执行”。当前完整新训练READY计数为0；这不是GPU利用率结论，不能把分配卡数或候选数当busy。

五项优先路线为Qwen财政表格、InternVL与OneVision多事件图表、VGGT室内RGB-D、SDXL固定图像颜色任务。财政已有海光8卡5次更新，VGGT已有单卡两场景25步及恢复，证明可行性但不授权重放旧请求；VGGT八对原UNKNOWN必须原位对账。T03/T04完整模型/initial、SDXL数值失败等缺项全部显式保留。未按“凑5READY”伪造资产或训练结果；各下一步均给root派发。

inspect_candidates.py --repo G 只核公开本地证据pin，输出五候选/下一阻塞，不连接服务器。metadata快照可陈旧，真实GPU派发仍必须fresh资源、当前实际文件、原请求去重及完整sourceclaim；普通Git/status/Slurm由root负责。无未授权隧道、拷贝或下载器。

prepare_vlm.py 是可执行的公共数据适配入口，支持T01/T02/T03/T04/T05。消费真实layout-v2 dataset及其精确manifest pin，拒绝旧视觉FAIL-v1；按id将train72完整user条件与对应gold连接为assistant `FINAL: answer`，dev36保持只有user，不读取dev gold或任何blind输入、标签、媒体。输出两份私有Swift JSONL与data-binding-private.json，18媒体仅只读SHA，生成明确Linux target_data路径的部署清单，不自行复制到目标，不假目标存在。

调用参数：--case T03.sft --dataset ACTUAL_LOCAL_DATASET --manifest-sha ACTUAL_SHA --manifest-bytes ACTUAL_BYTES --target-data /public/home/.../NEW_SCOPE/data --output NEW_IGNORED_LOCAL_OUTPUT。target_data用PurePosixPath，避免Windows误判Linux绝对路径。输出目录必须不存在。原样本顺序、完整图/问题、source group、labels mask责任和原生EOS边界保留，不手添特殊EOS、不裁剪、不shuffle。

此输出只是下一native CPU和部署的输入。目标存在false、actualencode0、真实有效监督tokens UNKNOWN，训练/GPU/Model GO false。最大context分别T01/03/04/05=4096、T02=512，只声明原限制，不截短；原模板、视觉冻结、LoRA、BF16、原8卡batch1/acc4/有效32和checkpoint/RNG恢复均由原统一训练框架及新独立资格负责。不得继承COCO64/Treasury的编码指纹冒新144数据native已过。

USGS新分组按已封声明保持train/dev/blind及事件去重，blind不作训练。原FiscalData、TUM、COCO来源和许可限制在候选卡中分别列出；COCO媒体与caption权利不能混称，已有源/下载/处理资格不代表可对外分发或效果已验。原1125财政数值、USGS快照、TUM原RGB-D和既有缓存优先复用，不重复取源。未来扩到其余生成/几何模型时沿各自原缓存接口，不把VLM JSONL强塞到视频、动作或材质头。

评估指标应分别记录数据资产与去重、原生输入资格、实际optimizer updates/有效tokens、全失败分母、恢复、任务效果及目标实测busy。旧50题/图表单域微增不支持领域能力晋级。训练/开发和已暴露盲组必须标EXPOSED；新盲组仅在固定checkpoint全预测封存和独审后首次开gold。本文与代码未打开任何blind或旧预测，未派发GPU。
