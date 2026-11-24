# O01 设备持有者权限诊断

原 `.2292` 在八卡和七卡两点计算计数增长后，进入见证前的设备持有者检查时遇到 `/proc/<pid>/fd` 的 `errno 13`。未执行见证。原独立 `.2291` 的 224 个 cgroup-v1 样本记录主机内存峰值 34,012,712,960 字节，三项 OOM 相关计数保持为零。原错误只记录了异常类型和路径类别，无法区分目录枚举与链接读取。

`derive.overlay(members)` 只接受两份已用于 `.2292` 的确切运行时源码，返回新的 `linux_owner.py`、`supervisor.py` 和 `fd_permission.py`。它在原容器、原监督器任务、原授权视图下记录 `process_stat`、`fd_enumeration` 或 `fd_readlink`，并区分通过完整进程身份绑定的保底子进程、监督器自身、同编号 Slurm 步骤、同 UID 的其他步骤和身份未确定。分类仅在已有权限错误发生后执行。任何 `PermissionError` 都会阻止当前交接，DRM、KFD 和同 UID FD 全范围检查仍按原入口执行。

未来采用方须先为新认领渲染并重新封存运行时、HOST 安装、作用域清单。调用者从原同一运行目录只读核验终态 SHA、已实证编号和所有者，再用 `fd_permission.terminal_diagnostic()` 比较 request、step、scope SHA 和 runtime SHA。可对外发布阶段、操作、errno、进程关系、KFD 成员布尔值和失败门禁，进程号、FD 路径、cgroup 原文留在私有证据。HOST 对相同 PID namespace 的独立验证可用于确认容器可见性差异，不能覆盖容器扫描拒绝或作为交接准入。若未来要更换检查边界，须在同节点、同启动实例、精确步骤下证明相同的跨进程 FD 与 KFD/PCI 覆盖，并验证拒绝场景。

本目录只包含源码和本机回归。未派发目标步骤，未使用 `.2296` 部署诊断，未读取或变更其释放回执。
