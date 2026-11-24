# Gemma VizWiz 原名义 25 步下的两步见证

本包复用已发布 TRAINING_SCOPE 与 build_view.py，原样输出经验证的 800 行训练视图。视图为 345,722 字节，SHA-256 为 cebeecad13db010a06f2ae7ef48260c6b26cc5ad42ea4993f6b8a60849ce1ad4。没有重写训练行、媒体路径或已发布物理顺序。衍生 lineage 只记录已发布视图自身的逐行 SHA 和两个源流身份，不能说明 Swift 实际抽样顺序。

渲染器按原 fresh-domain callback 的完整 SHA 生成新插件。保留 8 个本地 HIP rank、原始 25 步调度及每 rank 100 个 nominal microbatch，保留 batch 1、累积 4、训练 shuffle、原 476 个 AdamW 槽位验证及目标 fresh initialization。实际步骤 2 完成后停止，要求每 rank 有 8 次原生前向和 loss、两次优化器计数证据，并将旧训练检查点门替换为拒绝。此包不采用旧 initial.pt。

实际运行前须独立采用本次训练使用范围，接合已核实的 fresh-domain worker 和 supervisor。两个原 Train-only 媒体根应以不改路径的只读方式挂载。target 的零 GRES 原生索引阶段调用 derive_index，须传入实际旧项目已核验且设为 train 的 gemma3_vision 原 512 capped template。该阶段核验两份原媒体 bound manifest 和所有 12 张实际挂载照片，通过真实 encode/collator 算出 input_ids、labels 和转为 BF16 的 pixel_values 签名。回调在实际前向再次计算同一签名，记录命中的原始 source-line SHA，并拒绝缺字段、冲突或未知张量。相同源行的重复物理位置无法通过内容区分，回执只是已消耗输入的源行身份证据。

`prepare.py view` 生成私有训练视图和其衍生 lineage，`prepare.py render` 渲染新插件。部署时将 gemma_vizwiz_two_updates_inputs.py 安装于隔离框架的 vlm/runtime，给 witness config 绑定实际 lineage 和真实原生 index 文件的 SHA；数据实际可读性、索引、标签传递和最终 native scheduler/optimizer 仍由独立目标验证。本轮只做本地源码及真实冻结视图重算，未落盘训练视图、读取目标挂载、调用原生算子、执行 GPU 或训练。
