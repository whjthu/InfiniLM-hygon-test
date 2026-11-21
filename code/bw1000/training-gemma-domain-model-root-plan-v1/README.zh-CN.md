# Gemma domain 显式原模型目录绑定

这是新的 source-only 小修复，不修改旧task、旧工具、通用VLM默认、原配方或396框架内六份overlay。无远端、Slurm、GPU、权重正文/初始LoRA读取，所有GO=false，READY新增0。

**更正前一轮判断**：MODEL_FILES.json 的 `target_root` 是传输父根。原16文件已独审receipt的实际 `target_model_directory` 为其下 `gemma-3-4b-it`，并非父根本身。不能仅由target_root断言旧模型目录缺basename。本包以原独审receipt为权威，同时要求它等于 `MODEL_FILES.target_root / basename(MODEL_FILES.source_root)`，并核16成员的bytes/SHA完全一致；矛盾即拒绝。没有重验目标文件当前存在性。

原builder将传入models再拼basename。若根传的是已含basename的实际目录，会重复拼接；`verify_model` 又要求两臂required_assets严格等于原独审实际目录。因此新Gemma builder显式把两臂唯一`--model`值和`required_assets.model_directory`都改成上述已核目录。`models`参数及其原MODELS_DIR记录仍保留兼容；建议根仍传原MODEL_FILES父根，避免把legacy环境元数据写成另一层。无新模型bind/alias/复制。

实际交付为 `deltas/tools/prepare_gemma_domain_pair.py`。`build_delta.py --repo REPO --output NEW_FILE` 可从固定旧4218B builder重现该5910B新源，严格只接受旧SHA与三个唯一插入锚点。**安装器与训练worker必须在同一新scope使用此新builder**：原 `install_gemma_domain_framework.py` 和 `run_gemma_domain_pair.py` 都以 `from prepare_gemma_domain_pair import candidate` 导入；在根的新隔离源码清单里只替换这个tools模块，并保持两调用的repo相同。不可只改生成过的两个plan，否则worker重算会拒绝。旧tools/task/manifest保持历史身份；新scope外层须锁本包MF、发布commit和builder新SHA。

数据及396框架安装仍走已冻结入口；.511原344B producer和800视图本轮已由根实际核验。本包不重复CPU72，不做目标拷贝。原initial审批/目标原初始化、实际Slurm/PCI/KFD/端口、真实rank sampler、25+5与恢复仍独立，不能因本机计划9项通过晋级训练或正式17。

本机验证命令（需全新忽略输出目录）：

```
python -B -X utf8 code/bw1000/training-gemma-domain-model-root-plan-v1/check_CPU_once.py --repo . --output local-only/NEW_GEMMA_PLAN_CHECK
```

最终9/9：真实双清单一致、两臂argv/assets一致、重现旧double-basename、除两模型字段外与同caller原计划逐对象相等、正确原调用仅legacy MODELS_DIR caller记录不同、六overlay逐bytes相同、原verify_model真实metadata前缀接受、合成根冲突拒绝、旧builder未变。原verify_model的权重stat循环未执行；POSIX目录比较在本机采用PurePosixPath，不将Windows斜杠语义冒目标Linux行为。两次先前测试失败也保留：首次错误使用Windows Path测试Linux前缀；第二次未区分不同models实参的MODELS_DIR记录。修正均在测试器，新delta字节始终不变。
