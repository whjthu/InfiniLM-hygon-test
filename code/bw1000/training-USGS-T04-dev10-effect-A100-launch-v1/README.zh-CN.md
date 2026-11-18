# USGS dev10 A100 单次执行候选

本包包含冻结18文件core精确副本；新增普通SSH安装/父supervisor/身份入口，数学、原dev输入、cap256、FINAL canonicalizer和评分器不改。原domain FAIL仍保留。

实际执行原RSS-v2 driver SHA2307123的身份、VmRSS分类/采样、Popen held清理及registry清理函数；四个qualified源码逐字节复用，sampled RSS阈值由本调用方显式设64GiB（不是硬限制或峰值证明）。不是仅在说明里引用旧driver。

固定GPU0的原UUID/PCI；原真实卡为A100-SXM4-40GB，total40960MiB，pre/post free>=38000只作准入。当前inventory、compute PID、同UID selected nvidia minor/KFD FD、固定port29927必须全核；占用/未知立即失败，不杀别的进程、不改选卡、不填busy。CUDA_VISIBLE_DEVICES只该UUID，4CPU排除0/1。GO前同父monotonic snapshot age<=60s。

唯一新namespace见INSTALL_PLAN.json；远端namespace和本地LABEL-original-dispatch任一存在不得重放。local epoch从submit main第一行开始，包含校验/Git/read/upload。900全链只一epoch：native820、supervisor850、held/registry880、alarm890、client895；所有调用按余量缩水。原状态未知时不重试、不新服务/路由、不中途重置epoch。

worker run/native必须起初不存在，held child在确认PID/PGID/startticks/PPID/UID/host/boot与父source后才exec原core。原original400不重封/复制，仅逐source pin读核；模型只原路径+qualified原model inventory/stat连续性。

当前GO全false，0实际GPU/SSH/Git/encode。Root review采纳后执行seal.py --adopt（仅本新包8个GO、两层MF和INSTALL_PLAN coreMF链刷新；原兄弟core18与旧运行目录不改），verify_package.py后普通发布，再仅一次：
```text
FIXED_LOCAL_PYTHON -B -X utf8 submit_once.py --publication ACTUAL_40HEX_PUBLICATION --root-go
```
该命令仅用root原ordinarySSH argv；不读取密钥/config正文，不显示endpoint/原stderr。SSH255/timeout且无对应request终态时保UNKNOWN、保所有原件，不重调。

A100不部署/打开dev或blind gold、不执行scorer。真实30预测及generation-finalized全部封存，且独立terminal/post确认后，root另行最小有界收回预测（完整private），在L原dataset调用core/score_after.py；它先核30实际输出才开dev labels。此阶段不把待传输当已许可/已评分，不做新模型/数据生成。
