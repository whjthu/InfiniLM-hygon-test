# Gemma .511 后的训练视图落位

本包只补已合格 VIEW 到隔离训练 RUN 的字节接线。所有 GO=false，目标调用、编码、optimizer 更新和新增 READY 均为 0；不触碰原初始 LoRA，也不提交 Slurm。

原 .511 独审证明 72 条训练题、12 图，144 次 encode、72 次 collator、最大长度332、72条合计有效监督token615。800曝光是重复训练视图，不是800个独立样本；其真实监督token总量及各rank loader尚未观测。

现有 `tools/install_gemma_domain_framework.py` 生成396文件、两臂命令及安装收据，却不复制视图。`tools/run_gemma_domain_pair.py` 要求训练 RUN 内 `view/train800-swift.jsonl`。新 `stage_view.py` 固定从已stage的原VIEW读602794B JSONL和8196B ledger，核全文SHA、稳定文件身份、800完整行，独占写入训练RUN。已有intent、arm目录、view或本次attempt即拒绝；中途失败不覆盖、不重播。部署方须固定本包发布提交及SOURCE_FILES.json SHA，再调用；本包不自带远端传输。

目标次序（全部由根协调者执行，不能把本机计划当目标终态）：

1. 在已授权 HOST 核原训练RUN是否存在、是否有旧intent/实际句柄；原 `.511` 已终态，不能重跑CPU72。
2. 用现有安装器从原393源建立隔离396训练RUN。固定安装树 `6221bf33ebc85c180ace49273b67c709e007d4e5`；同一RUN若已部分存在先对账，不覆盖或换名绕过。
3. 根采纳并发布本包后，在HOST调用 `python3 -B <本包>/stage_view.py --install-target`，原30秒内只落位两文件。原CPU数据/媒体不重复传输。
4. 使用已冻结 `training-gemma-native511-handoff-v1/handoff.py --independent-report <1218B原报告> --install-target`，复制原344B producer收据，原SHA `3832af1aef87b88149d955417ad39f665c50fe957a5f91caaf42f44bbc992e80`，不能重新序列化。此步骤也在GPU intent之前。
5. **仍有两项真实阻塞**：原119430531B初始LoRA export记录实际自动审批REJECTED，不能重传/换名/seed替代；且共同recipe把模型定位为 `models_root/gemma-3-4b-it`，原16文件实际target根为 `.../training-gemma-original-model-v1`。现controller只把该根原路径bind，不提供basename视图。新GPU scope可审查一个只读container映射到 `/gemma-models/gemma-3-4b-it` 并设 `models_root=/gemma-models`；这需要controller的新精确bind，尚未实现/验证，不得直接用现命令宣称可跑。
6. 上述初值及模型映射实际闭合后，根创建新原始GPU intent，fresh parent1103606/node f13r1n12、8DCU/KFD/PCI/port准入，再调用原 `tools/control_gemma_domain_slurm.py RUN --repo REPO --intent-sha SHA`。不占用f13r1n08。两臂为continuous25与checkpoint20后新进程5更新，共30真实更新；checkpoint10/20/25、8rank/160尾部microbatch配对、恢复数值与真实sampler资格均仍需目标实测。正式17/性能/领域效果不晋级。

本机新增证据：真实800目录读取在 `resolve` 即 WinError5，0正文，非自动审核拒绝，未重试该目录。第一次合成IO6项中成功分支失败且未保存why，原记录保留；仅加错误诊断后的新合成目录6项通过，不能据此宣称首次失败已归因或真实copy合格。两次结果和首次权限失败都在SOURCE_PROOF中固定。没有重跑旧native资格/读取dev或blind。

仅检查计划：`python -B stage_view.py`。合成测试入口为 `check_CPU_once.py --repo REPO --synthetic-only --output NEW_PRIVATE_DIR`；真实模式仍依赖原目录读权限，不在本轮重试。SOURCE_PROOF记录明确的原源码、许可数据资产与测试pin，原USGS数据的来源/分组/版权说明沿原冻结数据资格，本包不创建新数据。
