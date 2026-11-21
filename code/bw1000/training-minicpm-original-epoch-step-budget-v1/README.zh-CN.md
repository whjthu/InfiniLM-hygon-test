# MiniCPM 原 epoch：实际 Slurm argv 预算修补

本包只准备已发布 T05 controller 的两个函数增量。原 `tools/control_minicpm_usgs_pair_target.py` 15681B / `750d54b23f44e670cc152a762a9e977cd72ea88399ecd17bc3f65b800d800bf0` 已冻结于 b7b88058，并在14文件清单中精确绑定，**原源码、清单和 claim 均不修改**。不新增训练 claim，没有可运行远端 launcher，不是新的训练或 READY。

## 已复现的问题

原 `execute_step` 会把 GPU client timeout 缩为 `2100-used`，但 `step_command` 总是 `srun -t35`。例如 clock fixture used=1910 在原 `used<1980-60` 门禁内，client190秒，Slurm 请求仍2100秒。若仅按请求时间计算可延到原 epoch+4010，超过 whole2400；这是源码/argv反例，**没有声称实际发生过晚运行的 Slurm 作业**。容器内 timeout 在 apptainer 启动后才开始，不能用它证明容器启动前已经闭合。

原 post90秒还被 ceil 成 Slurm2分钟/120秒。[SchedMD 官方 srun 文档](https://slurm.schedmd.com/srun.html) 明确时间分辨率为分钟，秒数向上取整，因此这里改为向下取整的完整分钟；小于一分钟直接拒绝，绝不生成表示无限制的 `-t0`。

## 真正的接线

`step_budget.py` 的 `step_command/execute_step` 在原 controller 命名空间使用，不能独立当 launcher 执行。`assemble_candidate.assemble()`：

1. 硬核原发布源码 bytes/SHA。
2. 只替换上述两个函数，返回完整候选 bytes 和 proof；原文件不写入。
3. 逐 AST 断言其余原 controller 全部相同，包括原2400/2100/1980、worker、数据、模型和 keeper/cleanup 入口。

替换后的 `execute_step` **实际把**缩小后的 argv 交给 `subprocess.run`，与同一原 epoch 的 client timeout 同时保存到 phase-accounting metadata。为既有 `--immediate=5` 留5秒，不加任何预算。原终态 wait 最多24秒及 whole 剩余减9秒、原清理/独审链保持。

计时向下取整可能缩短阶段可用时间至多不足一分钟，post请求为1分钟。阶段不足时失败/不新step，不能扩大预算弥补。Slurm 的 KillWait/OverTimeLimit 仍依赖实际配置，此改动只修正**请求与 client 的接线**；不能取代真实终态、强 owner、清理和 post，也不能保证已退出。

## 本轮实测

`check_CPU_once.py` 已将这两个函数真实装入原 controller 的内存候选，截获其 subprocess argv/timeout，同时模拟 accounting。不是只测一个独立算式：

| clock fixture | 实际截获 `-t` | client timeout |
|---|---:|---:|
| GPU used30 | 34分钟 | 2070秒 |
| GPU used1910 | 3分钟 | 190秒 |
| pre used0 | 1分钟 | 65秒 |
| post used2100 | 1分钟 | 95秒 |

timeout 分支也截获同一3分钟/190秒，保留 CANCELLED/失败。拒绝 expired、NaN、infinity、负值、bool、零/不足1分钟/越过原上限等；16项新增边界通过。原 controller 5项测试随后针对**新候选**通过，没有重跑旧23 worker fixtures、CPU88或任何模型。

首测失败仅因 fixture 误把返回 result 与额外含 epoch 的磁盘记录直接比较。原 controller 本就保存 `dict(epoch=epoch,**result)`；已在新独占修正测试目录正确比较。首次失败原件保留，预算 helper 没改。全部测试均是 process/accounting doubles，实际 Slurm/GPU/native/model/optimizer=0。

## 根协调者最短下一动作

本包**尚未部署、尚未接入当前已发布 controller**。根可取 `assemble()` 返回的完整候选 bytes，写入自己的新版本 controller 路径。实际采纳前还必须为新版本更新 `SOURCE_MANIFEST` / `PUBLISHED_FILES` 的目录与精确源码身份清单，绑定新 controller 和所用 helper；不能让新 bytes 自称原 b7b88058/旧14文件清单。若把两个函数直接落到新文件，不必在目标再导入 assembler。

然后只用原 request 的已登记调度流程：先原 CPU88真实独审、原请求/步骤对账及资产 staging，后 fresh8卡/PCI/KFD/端口、唯一运行/原epoch、25+5、独立 terminal/post。当前原目标实际执行0、原 claim false；本机修补不替代这些条件。READY仍0，原30次更新只保计划分母，不据 fixture 报完成。

`check_CPU_once.py` 原结果目录已存在，勿重跑。raw模型、数据或检查点均未读取；没有网络、SSH、Slurm 或 Git 操作。
