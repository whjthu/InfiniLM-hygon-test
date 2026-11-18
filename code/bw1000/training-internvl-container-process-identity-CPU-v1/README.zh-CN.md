# 容器启动身份诊断（准备态）

仅固定1087656/a14r3n04、2CPU/2048MiB/0GRES，单120秒origin包含Git/source安装和SSH。Slurm实际60秒，容器握手最多20秒，最多400轮/总512条身份记录（先触任何上限则停止并明确原因），HOST95秒停止/105清理/113accounting/115硬退出；root本机持有SSH需115停止、118回收，UNKNOWN不重播。build_body不是新SSH服务。

固定已用absolute Apptainer、无--rocm、三visibility=-1及原只读bind；容器只运行`python -I -S -B inside_handshake.py`，入口剩余至少2.1秒才握手，随后约2秒退出。不会执行native64 worker，不读input/metadata/weights正文，不import Torch/SDK，不采busy或VmRSS。继承CPU mask后仅取2个实际CPU，实际资源以Slurm TRES为准。

只从held Popen PID与相对procfd stat建立初始ticks/PPID/PGID；子树仅沿已核`/proc/PID/task/PID/children`，不遍历全/proc。每项在有限status前核相同PID/ticks/PPID/PGID，status只解析Uid/Gid/State/TracerPid/NoNewPrivs。读取status文件会包含其它文本，但不解析/记录RSS；不把读取字节说成只包含这五字段。任意procUID/credentials不符或权限失败均先持久phase/身份/errno，再拒绝并清理；不借进程名授权、不sudo/改ACL、不豁免UID。

被捕获的拒绝可以形成diagnostic_complete且SlurmC0，但frontend_ready仍false；这仅表示诊断原件与held清理闭合。拒绝后的未观察后代覆盖UNKNOWN，不宣称全机器无进程，也不授予CPU-native/GPU资格。原358 FAIL不修改，不能据本次诊断反推原358具体PID/UID。

pre-registered read_terminal仅消费新原件，绑定实际sourceMF/context与Slurm/两级held；无新Slurm/Torch/设备调用。root须独立review、全部三GO显式采用后普通pub，才可一次dispatch。准备态仅`verify_package.py --prepared`，默认verify仍拒绝未采用。
