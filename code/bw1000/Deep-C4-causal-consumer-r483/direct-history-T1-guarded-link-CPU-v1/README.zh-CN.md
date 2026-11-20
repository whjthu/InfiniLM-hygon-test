# DeepSeek C4 候选探针的一次性链接

在 `1103606` 的 I03 节点 `f13r2n01` 使用 4 CPU、8 GiB、0 卡。复用原 `1103606.156` 产出的 `probe.o`，只将它对 `1103606.487` 候选库的依赖绑定到绝对路径，保留原编译器、链接顺序和 13 个探针入口。接收端先核两项上游 Slurm 终态、父作业所有权、当前步骤和唯一命名空间，再创建安装目录和启动意图。一次 300 秒预算贯穿 HOST、容器与链接验证。

根调度调用 `launch.py` 的 `main(publication, transport)`，依赖现有认证的 HOST Python 3.12 回调。保留正式启动回执中的 PID、起始 ticks 和原请求。目标端容器在执行前重核清单和所有已发布源码字节。运行先留下 `intent.json`，然后保存链接计划、尝试标记、工具日志和 `CPU-result.json`。已创建的命名空间一律对账。

独立接收原 Slurm 终态和输出的 SHA-256 后才能消费本轮结果。`CPU-result.json` 的通过表示只读 ELF 核验通过。实际设备测量仍须核对 `dladdr` 的提供方和七个原模型权重范围。本轮申请零卡，探针、权重和模型的设备资格单列。

归档原 `sacct -nP -o JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160` 回执后，用 `tools/read_deep_c4_direct_history_link_terminal.py --root <原HOST项目目录> --publication <实际派发提交> --accounting <原回执>` 核对。运行中或无记账行时保留原请求，终态失败仍记录结果可见性。成功时只读取原 CPU 链接回执、尝试、计划、上游 `.487` 回执和探针产物，保留各项 SHA-256。
