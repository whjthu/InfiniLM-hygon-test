# Kimi 欠满 prefill 合批对照

此包为 r8487 最终封存源；入口别名仍为 r8485。依赖沿用 `../kimi-matched-session/` 的既有封存源，逐 SHA 见 manifest。数学、DSO、加载和销毁路径不变，不安装或升级依赖。

唯一入口 `launch_Kimi_bounded_r8485.py` 会创建单次 intent、独立 watcher 和一个有限测试。已有 intent 时拒绝再次启动；观察超时只能跟踪原句柄。实际执行前须按新机器修改资源并重新封存，不能原样复用旧集群节点、端口或 Slurm hold。

单次加载包含前后 QA、历史 B16/M4 anchor，以及 O/W/W/O 四个真实到达 epoch；每条实际调用轨迹随后独立 full-logit CPU argmax 校验。O/W 都使用同 Tick broker，旧588仅作诊断。唯一干预为欠满 prefill 最多等待10ms；记录计划 offset、实际送达和迟到。用户 decode、TTFT、ITL及服务窗输出/卡分别报告；服务窗包含prefill，不能当纯decode吞吐。

CPU/TCP 合同不是GPU性能结果。测试结果需等待实际终态、完整token/EOS、独立audit与资源清理；不得预先宣称提速。
