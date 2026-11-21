# GLM TP16 fresh K/V 候选的 decode 入口

原候选用单请求、单 token 和已有前缀识别 decode。若完整 prompt 以一行结束，它的最后一个 prefill 块也符合这些条件。此版本从生成循环显式传入阶段，只有新的 decode C ABI 会在当前 owner 线程安装当前 Runtime 的标记。attention 仍继续检查原有形状和候选开关。原通用 forward 入口继续供 prefill 使用，包括有前缀的单 token 尾块。

`render.py` 校验已冻结的输入 SHA，然后输出派生的 u00 bridge、u15 attention、u17 prepare 和对应 Python caller。新的 C ABI 沿用原参数并调用原 forward 一次。出现错误时清除阶段标记，外层 Python 保留原失败处理。生成的 owner 仍通过已有库身份检查解析符号。

`check_local.py` 检查生成循环和实际 Python 绑定的阶段传递，并通过本机 C++ 编译器验证 bridge 和两个翻译单元之间的标记隔离。目标侧需要独立注册和重编 u00、u15、u17，并用真实依赖和产物更新 Native ELF 导出清单、加载绑定及发布清单后再做设备数值和性能验证。
