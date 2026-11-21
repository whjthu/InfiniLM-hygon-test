# DeepSeek 配对编译派发与对账

O01 待办登记了 `deep-inverse-output-rope-prefix-width-pair-cpu-v1`，绑定两份编译包、一个新的请求及固定 CPU 资源。协调者收到该节点的原始新鲜 PCI、KFD 和进程收据后，经现有只读发布检查，用 `tools/dispatch_deep_rope_prefix_width_pair_once.py` 的 `dispatch` 获取原子预约并使用既有认证传输。作用范围是两个 gfx936 翻译单元、4 CPU、8 GiB、300 秒和零卡。

目标控制器核对两份发布清单和原有宿主机脚本，复查当前编号步骤，随后通过同一零卡步骤运行两臂。正向结果同时绑定原 Slurm 终态、两份编译回执、两份对象及各自的依赖文件。任一原输出存在时进入对账路径。

传输断开后，根协调者可经 `original_argv(..., reconcile=True)` 请求只读对账。此路径只读取同一请求的 intent、实际 Slurm 记账和原输出。状态未明确时保留原预约。确认终态后按现有共享队列流程保存证据并归还节点。

本机测试使用真实发布源码及原子队列夹具，覆盖传输超时、重复调用、错误父作业、产物变更和在途对账。实际目标编译、Slurm 步、GPU 数值和时延仍待验证。
