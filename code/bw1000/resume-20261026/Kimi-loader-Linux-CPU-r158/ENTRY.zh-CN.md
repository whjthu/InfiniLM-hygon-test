# r158：Linux CPU独立TU资格入口

只准备源码与本机CPU smoke；未SSH、Git、Slurm或GPU启动。root负责在已有407/n20上安排**单进程、8 CPU、4GiB、≤3分钟**的CPU步骤，不触碰当前模型进程。`CPU_inside_r158.sh <全新绝对结果目录>`验证parent910407/CPU8/a14r2n20后进入CPU_entry；它不会自行提交Slurm。外层175s、Python内部总170s预算，超时/失败保留结果。

交付必需文件：本目录顶层的`.py/.sh/.cpp/.hpp`及manifest；不要上传`local-*-observed/`里的Windows可执行文件。`regression_r157.cpp`、`exact_geometry_contract_r157.cpp`和两个helper头是r157原样拷贝。

## 防止fixture常量跨编译单元传播

基线、候选、driver分别`-c`成三个object，**相同**`-std=c++17 -O3 -pthread -fno-lto`后单独link。无unity build、无LTO、无march特化。两条extern C入口接收driver运行时构造的shape/stride vector；helper wrapper看不到driver常量。原helper与候选分别在独立TU，不能把fixture具体shape经LTO折叠进基线而改变对照。

本实验仍不是Native共享库实际安装路径；纯C++组件不涉及H2D、state_dict、真实weight pagefault。最终Model结果仍UNKNOWN。

## 固定有界测试

- 真正共享global源：每bank只有一份packed `[3584,1536]`和scale `[3584,96]` U8，8线程rank0..7分别读各自列slice，所有线程引用同一`banks` vector，不是每owner复制一个source。
- warm：1个bank，源5,849,088B；8owner各4096次两shape，ABBA四次。
- rotating：64个已初始化bank，源374,341,632B（约357MiB），8owner各512次；按iteration轮换，ABBA四次。目的仅为降低同一source复用，**不宣称cold cache/冷盘**，不drop_caches。目标CPU LLC/NUMA未知，初始化已触页。
- destination共约2.8MiB、metadata很小；不存在整模型/GB缓存分配，远低于4GiB。但最终Linux MaxRSS以实际步骤记录为准。
- 每轮校验8owner最终结果/guards和原5项counter；额外跑9,606个通用stride/重叠字节合同，以及两shape×TP16所有rank的32个完整source/destination/counter合同。
- baseline/candidate每次顶层copy均做相同5个relaxed atomic，不以关闭统计制造收益。

输出`CPU-result-r158.json`含每条编译/执行命令、stderr、compiler version、source SHA、实际host/Slurm CPU、各trial wall、blocks/bytes、资格结果；失败同样写receipt。执行本地`--smoke`只是入口功能检查（1/4banks、128次），不能当Linux/BW1000结果。

## 可选下次加载计数，不接当前加载

`loader_work_observer_r158.py`与`loader-observer-review-only.diff`仅供review，不集成、不由CPU entry自动import。若下一次本来就需要Model加载，root可将每flush进程CPU time、minflt/majflt、inblock/outblock和`/proc/self/io`差分加到现有flush_profiles，保持原Native/同步/源所有权。

进程CPU包含所有8owner与sampler线程，可大于wall；majflt不是copy专属，read_bytes与rchar含义不同，mmap已缓存读取可能无read_bytes增量。observer自身两次采样wall也记录，proc-IO包含采样自身的读取开销。Native load inclusive不能据此精确分割为pure IO/copy/H2D；这些计数只用于判断下一步该profile哪类工作，不触发报告专用整模型重载。

r156 fullsave助手继续PARK。r9212d registry缓存与r157 row fastpath保持独立归因，本机r157约1.8×不能宣称Model快20分钟。
