# Gemma 目标训练数据部署前对账

`tools/stage_gemma_render72_target_data.py inspect --publication 21c0207c1c83106ff18e4573448bb699a1b899b2` 由现有认证通道在 HOST 上只读执行。它核验父作业的实际所有者、RUNNING 状态和 320 DCU 记账，再查原 CPU 标签和两个独立目录。出现旧步骤、旧目录、歧义或查询失败时保留原请求作对账。

统一恢复工单提供该预检的固定参数。若只看见部署目录，`inspect` 重新核对两份目标收据和 16 个安装文件。完整通过后返回 `STAGING_VERIFIED_CPU_PREREQUISITES_PENDING`，可以继续核对原 CPU 入口的其余前提。部分文件或本机测试收据仍需对账。

仅当现有派发账本也没有该范围未决的发送记录、并且 `inspect` 返回 `ORIGINAL_NAMESPACES_CLEAR` 时，对同一源提交使用 `receive --publication 21c0207c1c83106ff18e4573448bb699a1b899b2 --epoch <原HOST epoch>`。只传固定的 1,120,578 字节私有包，使用一个 120 秒预算。目标入口在读包前和验证包 SHA 后分别复核原 CPU 作业和目录，复用原先通过本地往返检查的独占 `install`。本次控制器只批准条件满足后的数据部署，原包的本地测试授权记录和训练入口身份保持有效。

目标 `stage-admission.json` 保留两次 Slurm 记账 SHA、原 CPU 请求和包身份。传输超时或中断后查询原目的目录及 `stage-intent.json`、`stage-outcome.json`、`stage-admission.json`，保留已写文件和失败分母。实地核实原 CPU 容器、处理器、框架和节点后，再按原请求执行 CPU audit。数据部署与 CPU audit 都不计入 5 项训练储备。
