# T09 原694两文件受控恢复候选

当前全部GO=false，尚无用户例外许可。`relay_once.py`不是原694重播，旧694始终UNKNOWN。原目标保持不变，只有before.pt7149635B和inputs.pt1853149B，共9002784B；仅不透明字节，不解码tensor/不Torch/不GPU。

唯一未来入口：固定rootPython运行本目录`relay_once.py --publication <pub> --source-manifest-sha256 <MF> --exception-receipt <G内精确已发布真实许可JSON> --exception-receipt-sha256 <SHA>`。当前CLAIM和EXCEPTION.template均不可准入；必须先根审/发布具体源码，用户明确授予本次UNKNOWN动作例外，再由根冻结真实许可记录并更新CLAIM三GO/许可路径SHA后普通发布。不得用模板冒真实许可，不能凭一般CPU/GPU授权绕过。

源只读原A10045c64 capture两个精确路径，复用已合格12个纯stdlib relay函数逐AST与nice helper原bytes。新receiver先核原target：正确文件复用，不同类型/大小/SHA立即拒绝；源只发送missing集合。写自有同FS partial→fsync→os.link无覆盖安装，EEXIST只可核同pin复用，绝不replace/rename/delete旧文件。新fixed control directory和原request不同，已存在的新control只能对账不能再执行。原694 receiver源码与owner仍缺，不冒原执行恢复。

单一300s从CLI入口定epoch，Git/许可/源/目标均缩水；260s停工作，保留原300 hardstop贯穿清理/写收据。每端2CPU/AS512MiB/nice仅降优先级，0Slurm。已有ordinarySSH/A100alias/Hygonhelper不变，不新service/poller。强Popen先登记，逐child清理，任何forcedkill/SSH255/stream错误均UNKNOWN，不重试。原bodyenvelopeSHA、ownedSSH、READY、host/boot/PID/startticks和原epoch均留小私有收据。后续只读对账须同host核knownPID，不用跨HOST absence推结束。

文件payload最多9002784B。gzip应用流上限10MiB；Windows压缩数据单buffer<=10MiB，复制/转发同时两个显式buffer+frame<=20971528B，不冒整个Python RSS硬上界。Windows绝不落payload，stderr只受限私有留存不打印。源payload读<=9002784，目标已存检查及final验证各<=9002784，写<=9002784；实际物理I/O/NFSquota/SSH网络wire未知。

已做必要temp.bin边界5正常6拒绝，先前Windows FD/pathctime失败原件保留，仅测试适配修正，生产Linux五字段稳定性函数原AST不动。测试没有NFS/目录fsync/LinuxO_NOFOLLOW资格，没有PT读取或网络。`verify_package.py`只校源闭包与AST。

不可消除的风险：未知旧694可能不遵守新lock并在新校验之后写入；no-replace只保证本方不覆盖，不能保证旧writer不再写。两个final不是原子二元事务，可能只完成一个，必须保完整失败分母。SSH关闭不代表远端终止，新UNKNOWN也停止。未来consumer仍要按原pin核验，文件恢复不授予数值/训练/GPU_READY。

新原件位置：A100`PLAN.source_control/{intent,outcome}.json`；Hygon`原target/PLAN.target_control_name/{intent,validation,outcome}.json`。root实际后先读取这些小JSON及本机exec-intent/READY/transport，独立核2分母/sourceMF/许可/identity；不要再读旧42/694缺目录装进度。
