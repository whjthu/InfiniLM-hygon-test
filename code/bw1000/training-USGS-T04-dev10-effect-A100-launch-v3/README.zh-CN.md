# USGS dev10 独立 launch-v3

v1/v2真实FAIL及30未观察分母完整保留，两个原devicepost仍UNKNOWN；prior公版精确pin在INSTALL_PLAN/CLAIM。不能借原失败或951/512B诊断当当前清场。新namespace/request，8GO均false。

唯一变更是已获root/Astra采纳的当前own SSH传输观测边界：运行时从本supervisor的host/boot/PID/startticks/UID核到唯一sameUID directPPID及上游两级root0链。前后proc/status身份和fdUID0/mode0500/inode稳定；comm仅辅助，绝不读取实际exe。原951/512B的旧PID不进入runtime资格。

只有该directparent真实FD目录PermissionError可留一个uninspected_transport_ancestor；任何另一个sameUID不可读、PID复用/祖先链或身份变化都拒绝。传输祖先不入ownedchild registry，不发signal，不sudo/newservice。status身份字段必须稳定，调度State可在非终态间变化；原status整页SHA只记录，不假稳定。

例外路径before/post明确all_UID_FD_complete=false、nontransport_UID_FD_complete=true、transport_FDs_inspected=false、exe_identity=UNKNOWN，scope_policy=scope_admission_satisfied_with_transport_exception。这些UID字段只指本任务UID1001，root祖先仅元数据。若FD实际全可读则保留count0完整ownUID观测分支，不能假称未检查。任何路径都不声称allFDclean/exclusiveGPU。legacycore准入布尔只表示声明的scopepolicy检查完成，实际admission并列这一解释和覆盖字段。release只表示此声明scope的清理/准入。

原5字段NVML/UUID/PCI/实际DeviceMinor映射/noapps/40GB total40960/free38000/port门禁不变。core18数学/FINAL/scorer、原RSS/held清理、900含Git/upload的单epoch、base一次load三LoRA30输出封存、gold/blind关闭均不变。

Root最终审后seal.py --adopt→verify→普通pub，再唯一固定localPython -B -X utf8 submit_once.py --publication ACTUALPUB --root-go。任何旧namespace/localdispatch存在或SSHunknown均reconcile，不重播。新source只做22项新增纯CPU污染边界与AST/MF；不重旧fixtures、不远端/GPU/权重/Git。
