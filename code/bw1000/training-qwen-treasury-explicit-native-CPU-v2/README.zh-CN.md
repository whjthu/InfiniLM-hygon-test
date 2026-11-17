# Treasury instruction 修订的 CPU 原生预处理

160 条训练输入保留原答案、图片和顺序，仅 53 条 signed_difference 用户消息追加已冻结的格式说明。新 JSONL 摘要为 `5a6a8f02dbfc77aa46487a0180000275771aefd56e3cc933ffdcf7c97496162e`。输入文件独立安装至原作业目录的 `inputs.jsonl`，图片复用原 72 张 train PNG。

`check_explicit_native_inputs.py` 与已合格原 checker 的唯一字节变化为 dataset SHA。它调用原 Qwen3-VL processor、模板、标签及视觉 mask 检查和不等长双样本 collator。模型处理器使用 load_model=False，RoPE 构造只使用 Torch meta 参数。目标新输入的实际预处理资格尚待本包首次运行。

资源为父作业 1087656 的 a01r4n14，8 CPU、12288 MiB、零 DCU。唯一 epoch 从 launcher 入口开始，总上限 600 秒，Slurm 上限 8 分钟，原生检查最多 450 秒。复用原 bounded terminal accounting helper，真实编号步骤和 terminal observations 均写入结果。未知执行保留原目录及请求，不能再次提交。

根协调者发布并安装本目录全部清单字节至 `R/results/training-qwen-treasury-explicit-native-CPU-v2/source`，将新 JSONL 安装至父目录并核对摘要，随后以发布 commit 和实际 CLAIM.json SHA 调用 `launch_once.py --publication COMMIT --claim-sha256 SHA256`。运行前核对本包、160 输入与旧版53处变化、72图、父作业成员、同节点步骤和精确 label 历史账务。launcher 独占创建 intent 后只调用一次 srun。实际执行环境复用原 v4 的只读挂载和版本检查。

CPU 输入合格之后，再建立独立真实训练和恢复认领。此前 CPU、训练、开发生成或新 prompt 检查的资格分别由各自原件证明。

本 v2 保留 v1 前置失败原件，不重提 v1。唯一额外路径许可为系统 /opt/hyhal 严格解析为 /usr/local/hyhal，且目标是非符号链接目录；以该真实目标只读挂载到容器 /opt/hyhal。其余项目目录仍严格拒绝符号链接。新 v2 仍需独立实际 CPU 原生预处理，不继承旧资格。
