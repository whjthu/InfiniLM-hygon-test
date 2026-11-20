# MiniCPM 8卡容器内25+5 worker

这是独立 target worker 源码，接已冻结的 MiniCPM 训练核心；HOST资产准备、Slurm提交、原终态记账和独立post由root的tools负责。源码本身不连接集群、不发Slurm。本轮没有实际目标执行。

`worker.py --config RUN/worker-config.json --config-sha256 <实际SHA>`。RUN为 `R/results/training-minicpm-usgs-exposure25-resume20-target-v1`；request固定为未执行的 `b88f477d9e6f4e7d9e251c7090463b4f`。精确字段见CONFIG_SCHEMA.json（它是说明，不能作为实际配置执行）。HOST独立CPU资产stage没有GPU epoch，worker仅绑定其完成收据SHA；随后GPU driver建立唯一2400秒原epoch。容器内两arm共用 `epoch+1980`，没有按arm重置；有界清理在原GPU2100上限前收口。示例全部GO=false，实际CPU88原终态与独审pin是硬前提。

worker核真实13core源、隔离framework393和已patch两恢复源、800view/ledger、CPU88原始独审元数据、原ASSETS和HOST两计划。独立重建计划并只把原command[0]规范化为 `RUN/framework/vlm/runtime/swift_entry.py`，逐command/env核对，不能返回deployed base绕过隔离源码。目标model10/initial/source88/media22的强校验沿HOSTstage；worker不会再编码88或把nativeCPU资格当训练。

设备流程复用Qwen25+5和MiniCPM diagnostic-v3：同hostboot/8PCI、HOST完整FD前态与容器kernelKFD范围、fresh45秒、64CPU、port29929、SDK8张实际PCI映射。容器不宣称同UID FD完整。子进程使用独立session，登记PID/startticks并跟踪后代，只清理新请求精确命令或其后代。清理失败仍单独尝试wait/reap并保存UNKNOWN；没有VmRSS缺失=0路径，也不把Slurm内存分配冒RSS峰。TERM不会二次打断清理收据，HOST仍必须最终核真实编号步和post。

连续分支需8rank各100micro/25update和10/20/25保存；恢复分支需各20micro/5update、完整14件checkpoint20与原生skip80。checkpoint库存校验args.json、optimizer/scheduler、adapter、trainerstate和8RNG全文件SHA；本worker不反序列化tensor，不冒独立saved25值审计。比较器沿原精确阈值，原151漂移FAIL不洗。任意partial/missing失败保全8rank、30计划分母和已有进展，不把postclean冒模型PASS。

23项新本机CPU边界已通过：时钟/授权/PCI/KFD/freshness/资源、定点entry修复、进程非零与创建失败、清理异常仍reap、14件库存缺RNG、已有输出拒绝、部分7micro/1update保留、Mini回调精确匹配。它们明确使用CPU小文件和process/ownership double，没有真实nativechild、Torch、模型或GPU。首次partial fixture漏continuous目录导致断言失败，修fixture路径后通过；原失败记录在proof，不修改worker数学以迁就fixture。

目标第一次运行后仍要独立消费worker原件、HOST实际terminal accounting、post强身份/KFD/ports。worker的passed只是本scope完整训练/恢复观察，正式17项数值、效果、三轮性能和服务资格始终另外验收。
