# T04 单模型有界分期资产接入候选

本包只准备代码。root_GO、transfer_GO、finalize_GO 均 false，无实际复制、远端读取、目标探测或卡占用。T03 原未知 receiver、待许可 Torch/version.py 和 sixshadow 均不进入范围。

PLAN 逐字节绑定既有 T04 原模型清单：18 文件 17,070,557,079 B，其中14份元数据16,050,303 B，4个完整 SHA 已知的 shard 为17,054,506,776 B。第0期传元数据，第1–18期各传一个最多1 GiB区间。第0期从relay main入口（含Git准入）独立epoch90 s，后续候选每期独立同源epoch7200 s，源/接收各2CPU、nice10、AS256MiB，排除CPU0/1；这不是GPU或busy指标。AS不是RSS硬限。

root 先普通发布、明确允许本期后，唯一前台调用：

```text
<fixed-python> -B -X utf8 code/bw1000/training-T04-checkpoint-bounded-chunks-v1/relay_once.py --publication <actual40hex> --source-manifest-sha <actual64hex> --phase 0 --nonce <unique32hex>
```

每期本机和两端均使用固定 phases/00…18；nonce不是重试后缀。目录或 partial 已有即拒绝。首次接收只 stat 已登记的一个 metadata-v2/llava/model 候选下18个确切 basename，若已有任何权重候选则输出未合格 HOLD，不覆盖、不广扫；未找到只说明这些路径不存在。新源/接收目录各持有 request/hostname/UID/boot/PID/startticks/epoch 的 intent、progress、outcome。

源对权重要求原 device/inode/size/mtime 完全相符，区间读用同一 FD，读前后和 path stat 稳定；元数据当期全文 SHA 必须吻合。源/目标区间 SHA 交叉核验，目标完整写入 fsync 后才以不覆盖的 link+unlink 提交 chunk。Windows 每次只桥接1MiB，无权重落盘；payload缓冲界≤8MiB，不冒进程RSS≤8MiB。SSH/内核/文件系统实际 wire 和物理readahead I/O未知。

任何SSH255、local强杀、partial、缺结果都保持UNKNOWN/FAIL，不能换nonce重投。`reconcile.py`只读原phase的≤64KiB小receipt和已知PID，注入原PLAN/CONTEXT/MODE与新的≤90秒READ_DEADLINE后由root既有通道单次执行，不读body、不提交作业。跨host观察始终live=None；同host才能按boot/startticks认进程。缺namespace不能推出未执行。root真正消费原两端终态后，为下一期发表≤64KiB的独立authority（case、phase、request_id、source_independently_verified、receiver_independently_verified、same_original_host_processes_absent、asset_plan_SHA256、endpoint_SHA256），将其精确path/bytes/SHA填入CLAIM.prior_phase_terminal_authority，才可打开下一期。relay实际git-show核该原件并匹配本机前期原intent。审批开关可使后续MF变化，但PLAN和endpoint精确bytes由初期asset-intent锁定，不允许热改。

所有19期完成后，另开纯服务器CPU/I/O scope，finalize_GO仍需root明确采纳。`finalize_file.py`每次只处理一个原basename，单epoch7200 s、2CPU、AS256MiB，核所有原phase PASS/source-validation和完整连续区间，再从持久chunk顺读并组装新的 model/<basename>。最终 stream whole SHA 必须等于原18清单之一；不得从range SHA推whole SHA。完整最终阶段额外读17,070,557,079 B、写同量，无新网络payload；最大单文件4,995,008,464 B。1GiB限制针对传输payload，最终服务器whole验证为单独声明的CPU/I/O工作。保留chunks，最大持久payload约34,141,114,158 B，不证明用户quota足够；不在本scope清理旧数据。root在未来独立publication核源码/CLAIM后注入 PLAN/CLAIM/CONTEXT/MEMBER 调用该有限脚本，当前没有final实际调用。

最后 `asset_gate.gate(PLAN, original_PLAN_SHA, original_endpoint_SHA)` 仅接受18份完整whole证据及其当前文件stat全部匹配，才能称完整模型文件字节身份合格。原4whole SHA资格、chunk传完或单个组装成功均不替代这一门槛。初始可训练参数、runtime、数值、实际训练与效果资格仍false。元数据源代码只作字节传送，不导入/执行。

本地必要检查：`check_contract_CPU.py` 只测range/分母/nonce/partial/跨host/强杀契约；`verify_package.py` 只测本包AST、MF和精确字节预算。二者不证明实际源stat、目标存在性、网络吞吐或任何GPU资格。

Phase0窄修：work alarm先安装原hard deadline的仅自身os._exit(124)兜底，再抛TimeoutError，失败save/stdout只可用原20秒尾部。relay登记held Popen后先核剩余再写，逐child kill异常互不跳过，wait/join同原epoch缩水；没有新poller或延时。
