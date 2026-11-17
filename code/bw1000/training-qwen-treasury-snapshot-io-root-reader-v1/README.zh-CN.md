# snapshot-io 原CPU门禁独立终态reader

仅消费 request63c2aad05ff6451495951f5686141a23、pub043a1d4b59661d621ec329b8bbfbbc3c06a77948 的原11源CPU scope。独立解析3条actualsacct、原intent/outcome/sourceMF/claim、每arm原stepintent和stdout/stderr，固定整个JSON对象11键，不从混合日志择取JSON。核CPU2/4GiB/0GRES、实际step/node/8PCI/fullFD/0holder/boot、同600秒epoch；stderr完整SHA/bytes/marker与告警数量分开记录。

本次已通过既有ordinarySSH helper唯一实际只读执行。341/342/170均C0，三个stderr各53B且marker1、srun限频警告0。原始bytes及完整结果只在local-only/training-qwen-treasury-snapshot-io-root-audit-v1，公版仅boundedsummary和SHA。reader不提交任务，不改原件，也不因异常重CPU/GPU；GPU_GO仍false，新GPUv3必须另冻结完整scope。
