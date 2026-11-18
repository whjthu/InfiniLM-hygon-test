# 单节点原EPERM元数据诊断候选

审阅后root已采纳root_GO/CPU_GO，其他GO=false；尚未连接/执行。只在1087656/a01r4n14运行1CPU256MiB、0GRES短步骤，分阶段读取原fresh_inventory路径，不调用collect，不读取gpu_busy_percent，不初始化GPU/SDK/模型，不改ACL或sudo。

先实际核fresh父归属/40nodes320DCU/账本成员/当前仅batch和extern、newlabel去重；未知步骤停止。整体90秒epoch在本机Git准入前确定。Slurm最低1分钟如实记录为60秒；worker硬20秒、held客户端25秒后TERM2+KILLreap3；独立验实际ElapsedRaw<=30。不能把调度器60秒说成30秒。当前ReqMem空字符串不判资源失败，严格核actualAllocTRES1CPU/1node/256MiB/0GRES。

root发布后运行run_once.py --publication <40hex> --request-id <fresh32hex>；cwdG和固定rootPython。入口实际gitshow验证原352/192失败公版SHA再允许SSH。新自有namespace排他；输出只小stage/path/errno/filename/tracebasename行号。裸EPERM无filename，原80失败位点仍UNKNOWN；新单node观察仅新场景证据，不自动外推原80或重采fleet。采样器、训练、busy都不会因诊断完成而PASS。

原80个PermissionError、0/640有效卡、0/1920有效点全部保留。本包不重旧sampler/原34e，也不换调用入口取得任何待许可Torch或数据事实。只有新增纯stdlib内存单位解析2正1拒绝检查，未跑原fixtures。
