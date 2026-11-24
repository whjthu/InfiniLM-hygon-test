# Pack16 与 05:44 普通补位的单节点交接

本包只协调 group 6 的 `1103606/f12r2n10/.2930`。普通适配器加载固定 SHA 的 05:44 coordinator 和实际 70 节点日历，在原 `released → reserve` 调用处对这个节点做短锁仲裁。其余节点直接使用原 `reserve`。Gemma group 11 不进入本包的仲裁。

## 本地检查

从仓库根目录运行 `python -B -X utf8 code/bw1000/zimage-square-rhs-pack16-ordinary-handoff-v1/check_local.py <不存在的私有回执路径>`。测试调用真实 `groups`、`refill`、普通 ticket 写入路径和 Pack16 原 HOST 的准入出口。将观察、发布预检和子进程发送替换为本地捕获，覆盖普通 ticket 在先、业务确认未发射、业务结果未知及业务实际终态加独立清场四类。不会建立生产 ticket。

## 实际承接

先确认该轮 group 6 只由本目录的 `ordinary.py` 负责本节点，旧 05:44 入口和已派发子进程均已退出该节点的范围。旧入口与日历源保持原样。无此独立确认时，保留现行普通任务。其他组照常。实际 predecessor 自然终态、同节点物理预检和 Pack16 原 HOST 资源门槛仍必须满足，预算到期本身不构成终态。

独立的 adoption 文件应包含本包 schema `bw1000.o01.pack16-ordinary-handoff-v1`、准确 parent/node/predecessor、group 6、`calendar_SHA256`、适配器 `coordinator_SHA256`，并在根实证旧路径已排空后设置 `sole_coordinator_confirmed` 和 `legacy_target_dispatch_drained`。文件还包含 `observed_epoch` 和 `valid_until_epoch`，按实际 SHA 传入。采用确认是上游真实证据，本包不会改变系统中的普通调用或确认旧进程已经离开。

1. 使用 `arbiter.py accept --scope ... --scope-sha256 ... --adoption ... --adoption-sha256 ...`。业务 scope 另加 `host_output`，必须是项目 results 下的独占 `O01-pack16-2930-<16 至 32 位十六进制>` 名称。原普通三个 ticket 槽中任一个已存在时拒绝业务。
2. 临发射时通过 `arbiter.py grant --accepted-sha256 ...` 独占取得一次发送凭证。只有另行授权的正式传输者可以将精确 `grant.json` 带到新 `pack16_host.py run`。传入 `--business-grant` 和 `--business-grant-sha256`，其余参数沿用原 HOST，`--target-output` 必须与凭证绑定。本适配器只替换原普通排除检查与绑定该凭证的回执写入。原启动、实际账单、计算前资源门和独立 post 继续生效。
3. 业务取消且从未签发发送凭证时，使用 `release-unsent`。发送凭证已签发但无法确认传输是否发射时，状态保持 UNKNOWN。回执缺失或凭证到期均不会释放，双方继续禁止重新派发。
4. 已提交的业务使用 `release-post` 交付原 HOST 终态和 `finish` 的独立清场收据，按实际 SHA 校验。收据要绑定同一业务凭证、scope、实际一 DCU 步骤和独立清场。只收到终态不回填。若 worker 结果缺失或无效，原 HOST `finish` 仍能在精确终态和独立清场成立时写出结构化的负审核。该回执允许资源回填，保留全部计划分母，组件计时资格为 false。`finish` 的退出码 1 本身不构成清场凭证。
5. 两种释放都会返回原 05:44 路径的 `--resume-pack16` argv，可当即用于原轮次该节点。此入口只将 `.2930` 的一条记录交给原 `refill`，继续使用原终态观察、预检、ticket 和子进程授权。它不重新处理同组其他七个节点。

本地 ledger 位于原 ordinary journal 中，以专用目录和节点级短锁保存接受、单次凭证和释放收据。锁不跨传输或调度等待。未持有凭证的候选 HOST 不会绕过原拒绝条件。
