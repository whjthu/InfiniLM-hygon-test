# Qwen T01 Swift 原生 V 投影布局探针

这是下次获准、采用相同八卡 BF16 T01 训练配方时可选用的外部 Swift callback。它只在 rank 1 挂接，最多观察四次实际 `Trainer.training_step`，择首个真实命中的 `decoder_06.v_proj`。命中要求原生 training forward 内 BF16 `[1,186,4096]`、冻结 `[1024,4096]` base 权重、无 bias 和同一物理 rank 设备。覆盖率记录原生 microbatch、root forward、叶子调用、未匹配形状、训练步内 root 之外的叶子调用和实际梯度 hook 次数。root 之外的调用原样执行。

命中时用同地址、同 stride 的输入 view 隔离这个冻结线性层的 dX。真实叶子仅调用一次，原结果和原梯度均交回 Swift。持有真实 dY 的暂时私有引用，使用本机锁定的 `upstream_layout.py` 读取布局。仅当原 `training_step` 正常返回、同一原生边同时交出 dY 和 dX，开展一次后向返回诊断。诊断调用使用 rank 本地 RNG 上下文，持续受同一任务 deadline 及最多 20 秒的进程内检查约束，保留所有诊断分母。正常训练没有额外 DDP collective，保持原 SDPA、LoRA、精度、梯度累积和优化器。地址、版本、形状、report 写入等诊断失败保留正常训练返回。设备调用若失去响应，仍由正式训练自身的外层绝对截止和所有权清场处理。

错过四次窗口、发现不支持的原 dY 或一次尝试完毕后，解除属于本 callback 的 hook。诊断结束后 Swift 保持当前训练段目标，不触发新 optimizer 调用。文件 `vproj-upstream-layout-rank-01.json` 写在当前新训练的原输出目录，含覆盖率和至多一个诊断，使用同目录无覆盖发布。旧 `.348` 的 archive/RNG、Math SDPA 强制策略、供应商偏好与其他投影政策均留在旧源。

## 使用现有 T01 入口

1. 在经审核的新 caller checkout 安装本目录，并以同一共享只读路径提供原 `training-qwen-native-vproj-upstream-layout-source-v1/SOURCE_FILES.json` 和其精确 `upstream_layout.py`。后者在 rank 1 被原字节校验后加载。
2. 在原 Swift argv 上调用 `swift_args.add_observer(argv, absolute_plugin_path)`，把绝对 `swift_plugin.py` 路径与 `qwen_native_vproj_upstream_layout` 追加到原 `--external_plugins` / `--callbacks` 参数组。该 list 扩展沿用现有 Qwen Swift caller 约定，保留其余原参数和 callback 次序。
3. 在新作业实际环境设置 `QWEN_VPROJ_LAYOUT_SOURCE` 为精确原布局源目录，`QWEN_VPROJ_LAYOUT_DEADLINE` 为新作业原有绝对截止的同一时间基准。继续使用当前项目正式 T01 调度与已有挂载。标准输出无原输入或原 tensor。

本包没有新增训练启动器或目标申请。固定本机 Python 以 `-I -S -B -X utf8 check_local.py -v` 验入口契约。测试使用本机模拟训练与加速器对象，覆盖原 Swift 参数与回调登记、连续训练无命中、原生前向异常时完整清理、梯度不支持和只读来源加载。
