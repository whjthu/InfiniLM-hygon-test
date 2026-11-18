# T03 InternVL USGS dev10 三臂开发效果源码

仅准备：同一原 initial504、已独审 fresh domain25、generic25，在已冻结 USGS dev10（2 图）完整原输入上产生 30 个真实开发诊断输出。此包没有运行、上传、读取 raw 题或 gold、修改 status；BW1100 OFF。旧 T03 原生25/resume、旧 T04 效果均不重跑。

原 actual dev 数据固定 native-template-CPU-v2/data；dev Swift 3794B/ce5d467c…，native identities 73407B/34ce51a7…，只引用原件，不重生成题/CPU encode 资格。运行时必要的真实推理 encode/collator 逐字段对照该已冻结 CPU 原输入 identity；不附加提示、不截短输入。404 框架固定 fresh-domain 原目录；原 InternVL HF、internvl_hf、SDPA/BF16、r16a32、504 参数及冻结语言 target regex 均核实。两个 checkpoint 用原独审 14 文件 manifest 完整 SHA 绑定，再安全读取并核 typed adapter==原 producer live fingerprint。仅加载一次底座，原生 AdapterRequest 切三臂，逐臂 504 tensors 与实际模型 torch.equal。基座 eval/frozen；每臂 seed42 重置 Python/NumPy/Torch CPU/CUDA；无 optimizer、训练或请求 KV 复用。

资源 1 张固定 A100 GPU2、4 CPU（排除0/1）、RSS 64 GiB 采样停止线，采样非硬限且 peak 为下界。整体 epoch 起于 root dispatch 首行，900s 不重置；worker820、controller850、清理880、alarm890、普通客户端895。bounded_eval 调用原已实跑 RSS-v2 helper，held child 身份后才 GO，保留自有 PID/startticks、fresh NVML UUID/PCI/minor/apps/memory/port、前后资源与明确 FD UNKNOWN；无 allFDclean/exclusive 声称。所有 GO=false；root 必须实际普通发布 claim/完整源码并采用后仅唯一次调用 build_body.make_body 与既有普通 transport。任何 timeout/未知终态只读原 scope，无重播。

前置独立 consumer 已实现：root 在原 scope 结束后使用本包 collect_original.py（注入实际 CONTEXT、60s READ_DEADLINE、单独采用的 READER_CLAIM），仅小文件原件/进程身份，无 GPU/Torch/weights/gold；stdout 私有封存。read_original.py 独立核源码、全部30槽、真实自然 EOS/cap、前后资源及清理；缺失/失败完整保留分母。score_after.py 先校验该独审 result bytes/SHA 和 source/epoch，再校验全部30预测封存，最后才读取原本地 dataset 的 dev gold。任何生成/consumer失败不开 gold。所有 parsefail/cap/nonnatural EOS 均计10分母错误，不丢槽；frozen scorer 不改；optional 单层 FINAL 同三臂，strictFINAL另报。配对差值和固定seed描述性 bootstrap95 保留，仅10题/2图开发诊断，非盲测/最终质量/非劣证明。

root 待审：GPU2 当前空闲与 admission需运行前新鲜验证；此包未实际验证 InternVL 的同底座三 adapter 推理，仅复用实际 generic Swift AdapterRequest 入口并以真实字段/target/504 guard failclosed；首次未知生命周期失败不得重播。原 T03 resume drift 仍 UNKNOWN；新两25已通过 metadata 工程独审不代表恢复或领域能力合格。weights/tokens/predictions/gold/激活仅 A100或本地私有，不提交 Git。
