# Pack16 零 GPU 源码安装

本包把已发布的一次 HOST/仲裁源码和冻结运行时闭包绑定到原 Pack16。常规包在目标共享主机原位核对 .2662 的五件运行时和 SCOPED_FILES.json，仅传 20 件公开或已发布源码及固定事实。六件原运行时均一致才允许建立新安装目录。另一个显式的 include 模式仅包含这六件有现成精确校验的源码，适用于确认原目录无法复用之后的另一次准备。

## 本地核对

`SOURCE_CLOSURE.json` 列出全部 26 件原件的路径、大小和 SHA。使用项目 Python 执行 `prepare.py <不存在的本机输出目录>` 生成二进制 `packet.json`、`remote-command.txt` 和 `packet-pins.json`，默认只准备，不连接。指定 `--stage-id` 可复现同一远端独占路径。随后用 `check_local.py <不存在的私有验证回执>` 核对本轮准备的包并执行完整本机临时目录测试。测试使用临时目录，不建普通 ticket 或业务 grant。

实际一次安装的传输由根协调者在通道交回、包核验和相应授权具备后调用。把 `remote-command.txt` 去掉末尾换行后作为现有已认证 SSH 的单一远端命令参数，`packet.json` 原样通过二进制 stdin。不要通过 PowerShell 文本管道传包。生成的命令使用共享主机原 `/usr/local/python3.12/bin/python3 -I -S -B`，外层 `timeout` 为 55 秒，终止宽限 5 秒。SSH 配置、目标别名和凭证不在包中。

## 实际回执的边界

- 安装器先校验包、成员路径和旧运行时，再独占创建 `incoming/O01-pack16-2930-host-install-<stage-id>`。全部已写源码逐件重读，实际加载冻结 HOST 和仲裁入口，检查真实参数帮助并构造明确不可派发的虚拟节点 argv。输出 `INSTALL_RESULT.json` 及其 SHA；没有调用 Slurm、GPU、Torch 或 hipprof。
- 已知的历史目标 `results/O01-pack16-deps-json-c0243062d1f2/launch-and-accounting.json` 可以按自身 SHA 和精确两层记录校验，和本机 `original-outcome.json` 传输 envelope 分别报告。未提供 `--existing-original-envelope` 时，envelope 项为 ABSENT，表示本次未提供或核对某个目标候选路径。无需复制或合成 envelope；缺该精确原件时，原 HOST 的完整 `frozen` 门保持阻塞。若独立确认目标已有可读原件，可在准备时传入其项目内精确路径，安装器只读按原 SHA 核验。
- 安装回执仅覆盖共享主机来源和安装几何。计算节点的 fresh admission、原前序真实终态、设备清场与资源、业务 grant、私有日历可读来源和普通 coordinator 的实际采用均由各自原路径核验。安装器不会改变现有普通任务，也不释放、持有或派发节点。
