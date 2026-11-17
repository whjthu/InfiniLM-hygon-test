# WDI 训练输入到统一框架的适配（仅源码）

本包只产生同一框架 Swift JSONL 数据，**不增加训练循环、不修改正式17配方、不触发模型加载**。`prepare_train.py` 仅固定读取原run的 `derived/train/inputs.jsonl`、`derived/train/labels-private.jsonl` 和已核train PNG，按ID联结，保原输入顺序；216槽或24图任一不完整就整体拒绝，不筛容易题、不补位。输出只有 `messages=[user,assistant]`、`images`；assistant为 `FINAL: <冻结答案>`，images严格重绑到新领域namespace内原train/media。输入文件不接受任何hidden gold/已有assistant，禁止dev/blind媒体或gold进入训练；blind文件从不打开。

执行必须给独立reader实际结果SHA；核其已消费原producer成功终态、train/dev标签对应合格且blind未打开，随后按其原件pins核train文件和图片。图片重定位仅改变路径，不改内容/SHA；结果记录dataset、输入、标签、顺序、24媒体与框架API身份。所有新文件保留忽略目录，不部署评估gold。

接口证据为当前common a68952c清单指向的两份实际源码：

- `code/bw1000/training-qwen-target-unified-framework-a68952c-v1/deltas/tools/hygon_training_recipe.py`，15906B，b218b9a6c1b5b80aaf6df5a73006b06c1529a30a77b1e4ed7b6a8a1d7a9b4f96。统一入口T01委托`hygon_vlm_recipe.main`。
- `code/bw1000/training-qwen-target-native-CPU-v1/deltas/tools/hygon_vlm_recipe.py`，16415B，71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2。`build_launch(case,original,swift,models,dataset,plugins,output,master_port,...)`和`--dataset`明确接受独立JSONL。现有Treasury native checker实证使用相同`template.encode(messages/images row)`接口，**其旧数值或native资格不继承给WDI**。

使用同一原生Qwen3-VL插件/template/collator，不手工构造token/mask/EOS、不拼接特殊终止token、不替换optimizer/trainer。未来native CPU资格必须实际验证完整216例未截断长度≤4096、262144像素、image/user/padding mask为-100、assistant标签与input IDs一致、原生EOS被真实监督、collator padding/attention/视觉grid/有限值；本适配器不冒充这些已过。生成评估仍另要求typed natural_EOS/cap_hit，不将SFT数据联结当解码资格。

领域case独立命名 `qwen3vl.wdi-macro-chart-sft-v1`。此包只将新数据接到当前共同`dataset`参数，**不生成训练launch、不借旧baseline callback或偷偷改T01 registry**。新case的原模型/processor/template/初始LoRA参数与fresh seed必须另行精确绑定；当前model_base_SHA256、initial_trainables_SHA256未有，保持null/blocked。8卡BF16、每卡batch1、accum4、rank16/alpha32/dropout.05等正式配置不改；未来新领域预算、性能基准、原生数值、连续25步/恢复和效果另立可比配方与认领，不能沿用旧160财政或正式A100效果分母。

未来本地唯一输入准备调用（不是训练）：`prepare_train.py --run <原WDIrun> --reader-result <真实reader结果> --reader-sha <SHA> --common-source <上述common源> --vlm-source <上述vlm源> --target-derived-root /refactor/results/<新WDI原数据namespace>/derived --output <新独占忽略目录>`。未执行；全部GO false。
