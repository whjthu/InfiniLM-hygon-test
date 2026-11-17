# 连续25步产物的三个固定开发对评分准备

本 scope 为原连续训练 `training-vggt-tum-continuous25-recovery-v1` 的独立开发评分。沿用已有 Cabinet 1 对、Teddy 2 对，开发集清单 SHA256 为 `6a0490db52ea7120766b3741b9dc788533abb81b8d2006ec7e8a2535fbd49a88`。两臂用同一 resident 原模型，先评原始 head，再加载本次连续25步 head。原 `score_vggt_tum_pair.py` 的相对位姿、方向平移误差、AUC@5/15/30 和失败分母逻辑全部复用，只在新 scope 渲染独立源码以绑定实际新 head 文件身份。没有使用 household。

已得到的连续训练产物身份为文件 SHA `522483d98b126de0dcf00825cb32751d8bba7ca98a612d9d098a066b97c951e9`、模块 SHA `28d0b01564e01d59b3cbb2b20b43b4e5c5f69dbcaa042a1e0c5bf714f51c88ed`，stat 为 `[864719552,10703273036,1794875418000000000]`，continuous native-result SHA 为 `ee8ce9936e667173aa220b766fb10a601106d6e4f4e6d2bbd27d364b2675cc46`。唯一 head 路径为原训练根下 `arms/continuous/train/camera_head.pt`，不把恢复臂的 head 用作评估候选。

`training_guard.py` 的主机预检要求原 request `fd7afad15b594969b8e9273b939f1fd6`、原发布 commit、GPU `1087656.321`、before `1087656.320` 与实际 post 都有原件和一致的实时终态 accounting。独立 post 必须 `COMPLETED/0:0`，before/after 的 PCI 和 boot 身份相同，after 完整 FD 清单且 KFD holder 为零。原 continuous receipt 必须指向上述 result，完成 25 更新、20 计时步、双场景轮换和模型检查，head 必须在原 artifacts 中出现。预检还核其实际 stat 和完整文件 SHA。

本目录的 `training-binding.json` 已按根协调者提供的五份实际原件 SHA256 冻结，分别为 `outcome.json`、`arms/continuous/receipt.json`、`arms/continuous/native-result.json`、`before.json`、`after.json`。原发布 commit 为 `218d6e6f9ff6baebfb7f46d43b8db71c9d4feb5c`，GPU `.321` 和独立 post `.322` 均完成。schema 为 `vggt-continuous25-dev3-training-binding-v1`，精确字段由 `check_binding` 定义。`allow_recovery_failure=false`。若原恢复失败，必须在新发布 claim 明确作出独立评分决定，再允许为领域效果取证，输出继续保留 `original_recovery_passed=false`。开发评分成功不会改变原恢复结论。

`render_controller.controller(original_single_controller,binding)` 返回生成的独立主机脚本和规范化 binding 字节。它复用原单卡设备前后检查和终态 collector，通过固定 SHA 的自身 guard 在创建新运行根之前检查原训练。调用方需携带旧 `render_tum_dev_controller.py`、`render_tum_development_set.py` 精确源码。独立 scorer 用 `render_controller.scorer(original_scorer_bytes)` 生成。新目标 label 为 `training-vggt-tum-continuous25-dev3-v1`，parent `1087656`、节点 `f11r4n19`、1 DCU，预算和资源继承原合格开发 collector。

`submit_once.py` 仅在根协调者绑定终态并普通发布新 claim 后使用。新 claim 的 `sources` 需包括新 launcher/renderer/guard 和原 scorer、原三个 renderer/controller、`tum_vggt_pair.py`、`tum_development_set.py`、snapshot/assigned_kfd/slurm_accounting 依赖。必填 `generated_controller_sha`、`generated_scorer_sha`、`training_binding_sha`，以及原3对 `pair_payload_bytes`。原3对开发数据仅按清单传入。新源码准备阶段没有执行提交器。

本地 `check_contract_CPU.py` 使用显式合成的终态/状态字典验证门禁，拒绝 RUNNING、缺 post、错 head artifact、部分更新、错游标和未明确允许的原恢复失败。核对角误差与 AUC 的 AST 不变，并确认原模型只有一个构造和加载入口。CPU fixture 的 post 编号和发布后缀为测试值。随后使用实际 binding 在本地运行完整 `prepare()`，核验 39 个待供料文件及总 4,598,376 字节，生成最终 controller/scorer。目标端预检、GPU 分配和评分均待根协调者完成独立原件审核及普通发布后执行。三个原开发对继续是开发集，不能据此声明正式 CO3D 或独立泛化。
