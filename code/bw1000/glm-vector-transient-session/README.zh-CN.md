# GLM 瞬态 K/V 与向量加载组合验证

实际入口为 `launch_GLM_vector_transient_Model_r5694.py`，读取 `GLM-transient-Model-source-manifest-r5694.json`。该入口启动唯一 watcher，确认其就绪后才启动计算；不要另跑独立 watcher launcher。BW1000 已唯一启动 910376.463，不要重复提交。

私有库 a66a1f… 同时包含瞬态 K/V 与受限向量加载，默认均关闭。TP8×PP2、16 卡、B16/M8，同一次权重加载跑基线、瞬态、向量、组合及反向顺序对照；每项需完整 token/EOS/QA 通过。向量路径只覆盖实际算子 tokens=4/8，其余保留 scalar。profile 单独运行，性能来自 profile 关闭的阶段。组件倍率不能写成整模型收益。

源码以同目录 source manifest 及另附 launch/observer manifest 校验；父对象、私有 SDK、权重和冻结参考的路径仍需迁移核对。本快照没有携带 GPU 二进制，也未证明在另一集群可以原样构建运行。迁移后使用新执行身份，保留旧源及失败证据，先做完整资格验证再发布性能。
