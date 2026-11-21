# Z-Image native12 CPU v2：完整调度器交付闭包

新request `1268f832fa01473a8f523dc9b80d51ae`，新label/RUN `training-zimage-style18-native12-CPU-v2`，节点为root明确指定的同parent1087656下T07 `f11r4n14`。原n13与4ce20…请求保持原样，仍由根处理原队列和失败闭合；不得重播或释放其占用。

原v1只stage计划，load_scheduler因缺`qualified_sampler`导入失败，根报告未创建intent/submitted/编号sacct step。这是已知前置失败，不能称native12已调用，更不能把文件缺失单独当“不执行”的证明。本包不读取旧现场、不重做旧请求。

修复仅新包装的依赖交付：

- `sync_dispatch.py`精确复用fleet-qwen-projection-sync-depth-v1/dispatch.py，7679B/3c131766…；
- `qualified_sampler.py`精确复用fleet-lora-isolated-backfill-v1，6208B/09fc94b7…，其身份由原projection SOURCE_FILES确认；
- 同时交付原admission1423B/fc89becc…及ledger13304B/36361b92…；二者及sampler仅stdlib依赖。

HOST payload只读取本包的完整SOURCE_FILES成员和私有3601B CSV，不再在临发时从其它包补调度依赖。SOURCE_FILES覆盖pilot/launcher/三调度模块/ledger/worker/原生入口/metadata；接收器验证完整集合和每个pin。load_scheduler还验证实际三个模块的__file__来自本次独占目录。manifest不自包含自身，也不在pilot写循环自引用的MF常量；外层已发布source contract对完整payload含MF和CSV绑定。

`launch.ship(publication,snapshot,transport)`继续使用根现有本机shared-queue与fresh-admission工具，只有显式传入既有transport，无SSH/helper默认。共享工具只供本机队列准入，不是HOST导入依赖；不得把本包单独移到别处后绕过本机队列。HOST实际执行是`pilot.py`，它的调度依赖全部在交付闭包中。本包不调用ship/Slurm。

## 不变项

原12图和3601B CSV SHA f71b51fe…、source binding a06c29c8…、原生入口af9c8ce5…、40DiffSynth provider清单cc0aa379…不变，继续只读复用Gemma目标train/media；不复制/变换媒体、不读dev6或盲测。原native worker只换request与node，AST归一比较完全相同。controller command/scope/observation/reconcile/dispatch/receive六函数AST逐项等于原件；300秒原epoch、220秒worker截止、240秒Slurm上界与剩余缩水保持原实现，不因修import延时或重置。

先root采用CLAIM root_GO/CPU_GO并刷新MF后普通发布；GPU/training/Model/formal始终false。先fresh核f11r4n14独立清闲、成员关系、原queue hold和同label不存在，根再唯一调用ship。旧n13即便未放行也不借新请求释放它。CPU4/8GiB/0GRES，其余bind、ENV、原入口、实际12完整失败分母和后续独审保持原路径。

## 新的本机验证

```
python -B check_staging_import.py --CSV EXACT_PRIVATE_3601B_CSV --output NEW_PRIVATE_DIRECTORY
```

此测试只stage本包交付成员和同pin CSV，在新`python -I -S -B`进程调用**实际**pilot.load_scheduler。只在私有fixture模拟root采用两GO并重封fixture MF，生产CLAIM仍false。禁止任何subprocess调度调用；验证sync_dispatch、qualified_sampler、admission实际路径/全文SHA和纯node_map将n14映射到1087656/T07，不导入Torch/原生processor，不发GPU/Slurm/网络请求。结果由独立公版summary给receipt SHA，SOURCE_PROOF记录固定源码及验证范围。

通过只解决HOST启动闭包，不是目标native12/训练资格。最近的数据关卡仍是第一次真实native12输出像素/提示匹配；其后原生tokenizer/VAE、Turbo融合、全模型/初值/数值、连续更新与恢复保持独立，不能用此import PASS计READY。
