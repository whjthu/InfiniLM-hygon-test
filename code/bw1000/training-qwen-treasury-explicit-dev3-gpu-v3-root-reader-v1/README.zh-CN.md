# GPU-v3 原失败与清理独审

原3012fc14dd404fe39ffd12ce838a1154，源99c1e7fd。独立核24源、原intent/outcome、actual9记账、三GPU FAILED1和三post C0，pre/post完整快照的8PCI/boot/0KFD以及实际assigned PID/startticks清理检查源闭合。全部39输入/108visual/108text门禁通过，provider保存原14model文件17,545,907,231B校验；这不证明模型已经构造。三臂progress/raw/generation-outcome和worker-failure marker均缺，完整324missing保留。

首reader严格要求worker-failure.json存在，真实FileNotFoundError已保留在local-only/...gpu-v3-root-audit-v1（exactreader70122e）。修订仅把该缺失如实记录，第二只读独审完整通过，未重GPU。之后按根明确限定先读三个stdout尾≤8KB，再最后只读base两个147B/210B短日志并脱敏完整6行，未进入数据CPU域。

base精确致命日志为gpu.stderr第2行：LLVM ERROR: IO failure on output stream: No space left on device。没有Python异常/源栈；不能称Swift API错误，也不能凭26秒或缺失marker宣称模型未构造。其他两臂原终态已核，但不凭相似长度冒充其日志内容相同。实际失败的filesystem/cache路径仍未知。禁止GPUv4、重试或改名；Astra审核和完整HOST-main/容量/临时缓存准入必须先行。旧v1/v2各324missing、旧corrected-reader UNKNOWN保留。
