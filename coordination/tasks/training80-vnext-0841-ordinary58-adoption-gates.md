# 08:41 普通 58 台接管

候选来自 05:44 波次实际通过双点验算的 58 台，主父作业 24 台，替代父作业 34 台。原 12 台未完成接续、5 台既有保护、5 台单独接续均排除。f13r1n02 的独立零 GRES 清场继续用自己的认领。首准备时间为北京时间 08:41:03.651，首尾原预算为 08:56:03.651 和 09:24:39.022。

固定一小时观察的原第 880 和 930 点各有 640 个有效值。两个点均有同一组 17 台、136 卡报 0%，与 07:45 无 RUNNING 编号 GPU 步骤的节点一致，与本波 58 台无交集。其余 63 台平均 99.2421% 和 99.0496%。若其余卡保持该点实测值，达到 95% 至少还需恢复 14 台到满忙度。七台已证实桥接前退出节点即使全到 100%，全池也仅为 86.9031% 和 86.7516%。这些是两个脉冲的条件估计。原脉冲拆分 SHA256 为 `38ad51f3639e0b710e2ae6d1f294843d99c7881e05f8c5b4ac28981e432c0063`，07:45 调度原件 SHA256 为 `b0ab4c39957e57d15bd4ecd5b3f114b8644f7d2f4752ca83eb7edea7fb4ee5bc`。

## 本机只读

```powershell
$py = 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -B -X utf8 tools/run_training80_ordinary58_adoption_once.py check
```

入口走生产 `coordinator.py` 的 `calendar_scope` 与 `calendar58.validate`，独立复算 58 台和 16 组。私有请求计划、分组 roster 和原结果 SHA256 分别为 `96908d6b7b25e75e57a6756a12da273fe252a15e1935f18d6eace6c5d403152d`、`a9d84c05150dc61f1751e904cceb99c05672e7dfbefa9981fde683aff14b4c5e` 和 `56da79542a9bb1181385e66f68ac9523f32c323b3cd31cb73a530283a329c83f`。

上一波唯一运行者在 08:07:09.158 完成正式交回，原 wrapper 会话均已消费完成，按确切 launcher 名核验 `Win32_Process` 的匹配数为 0。私有退出回执 SHA256 为 `9d37a77db68c2162170aa871da58daa146dd703b6bb5b9e7152a829dc562c1c7`。组 5/14 的原 bridge UNKNOWN 与旧台账仍有效。正式交接入口再次读取进程清单。

## 真实交接后

当原固定小时独立读回确认 `.3088` 和 `.1182` 均有实际零 GRES 终态并释放，根协调者以实际结果填写私有 `local-only/training80-vnext-0841-ordinary58-operator-handoff-v1.json`，并将确切 SHA256 交给当前唯一协调者。接管只依据释放证据，利用率是否达标沿用监测原验收。

交接记录的固定字段为 `schema=training80.ordinary.vnext-operator-handoff.v1`、`owner=/root`、实际 `current_sole_coordinator`、带时区的 `issued_Beijing`、`wave=training80-vnext-20261125-0841-ordinary58`、`calendar_SHA256`、`group_roster_SHA256`、`group_request_plan_SHA256`、`legacy_owner_exit_SHA256`、`previous_ordinary_controller_released=true` 和 `sole_current_coordinator_authorized=true`。`observer_terminal` 记录原目录下独立 `independent-terminal-<read-id>.json` 的 `file` 和 `SHA256`。

`ledger_custody` 记录根协调者核对的库历史。在确认固定路径确实从未初始化、从未接管时，写入 `state=NEVER_INITIALIZED_AND_NEVER_ADOPTED`、`history_confirmed_by=/root`、`historical_local_validation_SHA256=fd3f7aec2b60e5efe544a3aa24bb9a659091a1731edd63a95cf73bfa8f86aa63`、固定库的 `fixed_ledger_absolute`（绝对路径且使用 `/`），以及 `affirmative_owner_history_reviewed=true`。如已存在可核对的固定库，写入 `state=REUSE_EXISTING_FIXED_LEDGER`、相同的 `history_confirmed_by` 和 `fixed_ledger_absolute`，另写已验证的 `ledger_device`、`ledger_inode` 和 `current_wave_never_adopted=true`。文件缺失本身不证明库历史。原路径和封条保留。

```powershell
& $py -B -X utf8 tools/run_training80_ordinary58_adoption_once.py check-handoff --handoff-sha $handoffSha
& $py -B -X utf8 tools/run_training80_ordinary58_adoption_once.py activate --handoff-sha $handoffSha
& $py -B -X utf8 tools/run_training80_ordinary58_adoption_once.py show-dispatch-argv --group 1 --adoption-sha $adoptionSha
```

第一条只读。第二条先写不可覆盖的接管意图，在历史确证全新时仅调用一次生产 `init-ledger`，核真实设备号和 inode 后写不可覆盖的接管回执，立即调用生产 `calendar58.require_adoption` 验证。任何中断保留原意图及所有实际库状态，后续先对账。第三条仅返回已绑定计划中真实请求号对应的生产 argv。实际派发仍须原候选全部终态、新鲜设备准入及生产发送门，沿用现有唯一协调权。
