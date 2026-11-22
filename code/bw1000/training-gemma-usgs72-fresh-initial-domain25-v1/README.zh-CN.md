Gemma USGS72：目标端新初值的独立领域配方

本包使用已经存在的 Gemma-3-4B-IT 16 个权重/元数据文件、已真实通过 .511 的 72 条 train QA/12 张图，以及 800 行冻结曝光视图。800 是重复曝光，不是新增独立样本。dev36/blind36 不进入本包。来源为既有 USGS 官方地震数值的本地派生图；不是 USGS 官方图。沿用原 source/alias/时空地理 closure 和 split，不重取数据、不重绘、不以答案筛题。原数据资格见 results/bw1000/2026-11-22/training-gemma-render72-stage-and-original-1103606-511-terminal.json，以及 results/bw1000/2026-11-21/training-gemma-render72-original-local-view-and-data-audit.json。USGS 来源/使用条件：https://earthquake.usgs.gov/fdsnws/event/1/ 、https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits 。模型继续使用已有、单独资格的 Google Gemma 权重及其条款，本包不下载或再分发权重。

这是一项明确的新领域实验。目标 PEFT 按 seed=42/data_seed=42 创建新 LoRA，实际捕获 8 rank 的初值 named SHA、rank0 新初值本地文件及各 rank 本地 RNG；观察前后 RNG 必须相同。它不读取原 A100 initial.pt，也不声称与其数值相同。原 A100 初值传输拒绝、原正式17配方和旧任务均保持原结论。本包不解决它们。不得把本包执行权限解释为模型或效果资格。

共同框架仍是原 a68952c 393 源的隔离396视图。实际旧396全体逐 SHA 验证，复制到新目录，覆盖6个已冻结 overlay；只有 continuous 初始化分支改变。原9个训练方法 AST 不变，resume witness/cursor/优化器更新门禁沿用原代码。common Swift 入口、Gemma model_type/template=gemma3_vision、max_length=512（不是4096）、BF16、SDPA、GC=false、LoRA r16/a32/dropout .05、语言476张量/29,802,496参数、冻结视觉/投影、AdamW/cosine/3%预热、bs1×8卡×accum4 保持。合法首步LR=0不误拒，完整窗口必须真有参数变化。

固定目标根：/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs72-fresh-initial-domain25-v1
固定 staging request：7b01c69d091a4230ad18b068a5dbf740
固定训练 pair request：247fa3682a114542ac3640fbc726aaae
目标仅建议 1103606/f13r1n12；根需确认当前队列/算子已清场，不能占用活跃 owner。没有独立 SSH、Slurm 或后台服务入口。

第一步是新 0-GRES CPU stage：建议同一目标节点/同一 DTK 容器挂载、2 CPU/8GiB、单300秒原epoch；来源包放在 RUN 之外，RUN 必须不存在。三个 GPU mask 都为 -1。使用与 .511 相同 DTK/site：source /opt/dtk/env.sh；PYTHONPATH=/minicpm-transformers:/swift-runtime/site。调用：

    python3 -B /绝对源码包/stage.py --epoch ORIGINAL_STAGE_EPOCH --deadline ORIGINAL_STAGE_DEADLINE

原样只读挂载旧 domain RUN、Gemma model 根、native72 data 根、运行时 /swift-runtime 和 /minicpm-transformers，以及合格 driver alias；新 RUN 为唯一可写数据根。stage 无 Torch/模型导入或反序列化，无 SDK/设备调用；会完整流式 SHA 核已存在模型16文件（约8.6GB读取，0权重复制），逐396源/72行→800行/12train PNG、344B CPU receipt，并记录真实文件 identity。预算不足就 FAIL，不续期。ROOT 已实查旧 manifest 92454B/7abc7f31…；不凭旧存在证明此次 stage 通过。

第二步由根的已合格8卡 supervisor 包装 worker.py。每次启动必须绑定 SOURCE_FILES.json、stage-complete.json、root run intent、fresh admission SHA；root负责 held PID/tree、RSS未知覆盖、实际 Slurm/accounting 和独立 post。worker 自己仅 exec 同一个 common Swift entry，不单独实现监控器。必须复用原合格容器（GPU --rocm）及只读 model/data/runtime挂载，PYTHONPATH 还含 NEW_RUN/framework。不能直接在登录 HOST 调用 worker 冒充 GPU 准入。

run intent JSON 必需字段：request_id=固定训练request，root=上述新根，parent='1103606'，node='f13r1n12'，epoch/deadline/native_deadline（有限、同一个<=2400秒epoch，native_deadline<=deadline-120），source_manifest={bytes,SHA256}，stage_complete={bytes,SHA256}，boot_id。不得有 initial_trainables。admission 是根真实独立同机原记录：同 request/node，pre step 为1103606.NUM且不同当前GPU step，8 PCI、boot_id、monotonic_ns，kfd_holder_count=0，full_fd_inventory_performed=true，master_port=29932，master_port_available=true。worker 会核<=45秒新鲜度，再核容器 KFD/DRM/HIP PCI与空端口。

    python3 -B /绝对源码包/worker.py continuous --run RUN_INTENT --run-sha EXACT_SHA --admission FRESH_PRE --admission-sha EXACT_SHA

首个实际目标仍是 continuous25：8rank各100 microbatches、合计800训练曝光、25真实更新；保存10/20/25原生检查点。step20每rank已完成80 micro，accum边界完整。checkpoint20原生14成员：adapter_model.safetensors、adapter_config.json、trainer_state.json、args.json、optimizer.pt、scheduler.pt、rng_state_0.pth…rng_state_7.pth。不存在这些成员不构造替代状态。

read_terminal.py 只产 metadata candidate，并流式稳定SHA核新checkpoint body，不反序列化、不导出tensor。建议独立2CPU/120秒/2GiB；它不是最终Slurm/cleanup审计。根须在 continuous 实际C0、8rank完整、强owner/tree清理和独立 post 后，生成 continuous-independent-terminal.json。其必需字段：passed/independent_passed/accounting_C0/owned_cleanup_passed=true、固定request、arm='continuous'、actual_updates=25、ranks_completed=8、microbatches_per_rank=100、checkpoint20_files=14原件{bytes,SHA256}。纯producer candidate不满足该门禁。

恢复是同一2400秒epoch内可选的第二个独占GPU阶段（最大5更新），根仍需新的实际8卡准入；run intent额外绑定 continuous_independent_terminal={bytes,SHA256}。只有原 continuous 独审通过才：

    python3 -B /绝对源码包/worker.py resumed --run RUN_INTENT --run-sha EXACT_SHA --admission FRESH_PRE --admission-sha EXACT_SHA

原 native Trainer 恢复 checkpoint20 的 adapter/optimizer/scheduler/8rank RNG和cursor；真实100→skip80→20 micro/rank/5更新。输出只到 resumed/train，不能覆盖 continuous。根可使用既有 tools/compare_gemma_domain_recovery.py 精确比较原连续尾部160个forward/loss/40个rank更新/adapter+optimizer+scheduler+RNG；尚未执行，不预称恢复PASS。连续25和恢复5是两个最多一次原生调用/两个GPU步骤，均失败保分母，UNKNOWN不重播。deadline不足就不启恢复；不延长旧epoch。原共有数据不变，不混成吞吐或正式配方结果。

权限初始全部false。根采纳、刷新MF且冻结后才能 stage；GPU前还必须根真实fresh资源准入。为保证 stage/gpu 的 source_manifest 一致，权限采用最终同一封版后执行，不在已stage后静默改源。任何重封需新明确阶段证据。本机 LOCAL_CHECKS.json 仅9项纯stdlib测试，不证明目标软件/LoRA值/GPU/训练。首次Windows fstat fixture失败留存于 local-only；修复仅测试fixture的Windows适配，目标Linux五字段核验没放宽。READY新增0；target stage、真实first25、恢复与效果均未过。
