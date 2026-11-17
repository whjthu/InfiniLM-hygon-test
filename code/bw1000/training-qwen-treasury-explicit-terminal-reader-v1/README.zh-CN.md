# 原 explicit Treasury 328 独立终态 reader

只消费原 df40e4… / pub29631a9… / epoch1794878306.5448935，固定327/328/329及已核 intent、launch、controller、launch-outcome SHA。先读真实 sacct 完整列，核actual CPU326原outcome/intent/log，核训练17源清单、新JSONL及8rank全20microbatch/5update/初始化和状态。逐checkpoint2/4/5流式hash14文件，保留42文件完整分母；不导入Torch、不加载checkpoint张量、不恢复模型。

root在原HOST以绝对Python运行 read_original.py --output 新private路径（默认--root为原RUN）。输出只打印passed和保存路径，完整private结果保留原件SHA、每rank输入hash与checkpointpins，不stdout打印原inputs/ownedchildren。reader异常只能修reader/对账原件，不能重Model。

模型结果和独立post分别核；CPU合格不能代替真实328独审。partial/missing/RUNNING保留原accounting和完整计划分母。记录初始504张量/43646976参数、空optimizer与RNG未恢复；记录update1→5adapter变化和所有rank状态一致。fullinputhash保留实际顺序，但旧输入control的逐rowID对齐尚未独立证明，不能自动宣布可比。冻结visual只证明callback记录requires_grad边界；不冒全部visual字节恒定。恢复、正式T01、三轮性能和盲测效果始终false。

CPU checker使用明确synthetic记录，只验证独立reader边界；不复用旧真实终态作为新实验进度。
