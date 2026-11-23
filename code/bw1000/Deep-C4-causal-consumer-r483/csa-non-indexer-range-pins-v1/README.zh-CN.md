# DeepSeek CSA 第 2 层非 Indexer 权重范围

原 23 项 checkpoint 头部已实查。此入口在 1194900 的 f12r3n17 上，以独立零卡 Slurm 步骤校验 16 项非 Indexer 权重值的 SHA-256。先核实际头部及 23 项范围间无重叠，再分段读取 16 项共 157,460,736 字节。七项 Indexer 范围继续沿用原请求的对账边界。原张量值只在实验侧读取，输出仅含小型校验值及计数。

在经实际账务确认的 1 CPU、1 GiB、零卡步骤中调用 `read_ranges.py --checkpoint CHECKPOINT --metadata METADATA --output NEW_PRIVATE_RESULT`。步骤使用 `CLAIM.json` 的唯一标签，90 秒 Slurm 上限及 45 秒读取预算。现有八卡保底需由发起者现场确认仍是原句柄，步骤完成后再次核实。

本结果只证明这 16 项范围的原字节身份。完整 23 项加载、实际 Native 枚举和 OFF/ON 设备数值比较继续依照已冻结计划验收。
