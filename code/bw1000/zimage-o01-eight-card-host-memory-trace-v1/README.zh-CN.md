# 未来 O01 八卡见证的宿主 RAM 取证

原 .2175 在 24 GiB 预算内运行 229 秒后由 Slurm 记录为 OUT_OF_MEMORY。原 .2185 对另一个普通单进程八卡保底 .2179 读到 cgroup v1 峰值 16540413952 B 和限额 17179869184 B。两种进程结构不同。建议下一次先固定 --mem=49152M 作为探测档，保留原节点、8 DCU、12 CPU、240 秒和数值合同。48 GiB 是失败申请量的两倍，真实混合峰和临界预算需重测。原 .2179 自然结束前不抢占。

## 接入步骤

1. 新见证使用新的 request、标签和目录，scope 增 host_memory_MiB、memory_observer_label，继续绑定新 scope 原字节、boot SHA、attested_entry_SHA256 和 runtime_SHA256。未来派发由根完成，本模块没有提交或取消入口。建议独立同节点零 GRES 观察步使用 1 CPU、256 MiB 和有界 Slurm 时限。保留足够实际 CPU，以便它与 GPU 步的 12 CPU 亲和性完全分离。
2. 在同一个原 epoch 内启动观察器。它独立读取 Slurm 实际任务和 cgroup，从已验证的目标 attested host owner 判定真实目标步。复用 .2185 的固定 cgroup v1/v2 读取与 PID 验证。输出目录独占，阶段文件目录由新监督器在原 run 下创建。两者使用同节点单调钟。

       python3 -I -S -B ram_trace.py --scope SCOPE --scope-sha256 SHA --owner ATTESTED_OWNER --reader ORIGINAL_OBSERVER_PY --attested-source ORIGINAL_NODE_ENTRY_PY --phases PHASE_DIR --out NEW_TELEMETRY

3. 未来监督器在 inventory 或任何 HIP 调用前调用 await_baseline(NEW_TELEMETRY / "ready.json", scope, scope_sha, step)。只有独立观察器已持久化第一个真实样本，而且实际 step memory hard limit 和可见祖先的最小硬限额与计划一致，才放行。等待继续使用原 epoch。原数值 sequencer 可在已有边界调用 record_phase(PHASE_DIR, scope, scope_sha, step, name)，见源码 PHASES，阶段记录采用同目录 no-replace 原子发布。建议在冷启动八个进程前写 cold_standbys_enter，八卡和七卡实测达到原门槛后各写 ready。探针和第八张恢复前后另记，最后在清理前后标记。
4. 根把独立 Slurm 会计和采样文件绑到原编号终态。每个采样先读阶段、再读 RAM counter、再读阶段，原始样本按秒独立持久化。峰值和事件可能是从 cgroup 建立起累计，跨阶段样本保留两个标签，短阶段可能没有独立快照。原 cgroup 或计数器消失时保留已写记录。消失和计数读取失败有自己的结果。单次采样不能推断 OOM 的最后一瞬，Slurm 状态仍独立核对。整个 GPU 步 cgroup 与后续普通保底各自对账。

所读数值为主机 RAM。没有 HBM 峰值或正式 GPU 性能测量。该模块未改历史包，也没有创建真实 Slurm 步。
