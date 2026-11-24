# Pack16 零卡依赖预检与终态复核

预检固定 `1103606/f12r2n10`，只允许与原八卡保底 `1103606.2480` 并行。根协调者先检查该保底仍实际运行、源码普通发布正本、原预检标签与 namespace 全空，再持久化唯一提交意图。`precheck.srun_argv(precheck.STAGE / 'precheck.py', source_sha)` 渲染 1 CPU、256 MiB、零 GRES、Slurm 最长 60 秒的步骤。`STAGE/runtime` 必须是已发布 pack16 原 `prepare.py` 产出的 5 源与清单，预检外层和容器内分别核相同字节。

预检前后都核 `.2480` 的 Slurm 标签、节点、资源、同 UID PID/startticks/cgroup/boot 与 `listpids`。DTK 容器沿用已验证 `--rocm --no-home` 和 `/training-runtime:ro`，清空三个设备选择环境变量。对 hipprof 与 HIP ELF 只核路径、SHA、架构、解释器和 `DT_NEEDED` 声明，并静态定位 Torch 包、扩展文件及声明版本。未执行 ELF 动态解析、未导入 Torch 或运行 hipprof，实际 provider 装载与 GPU 功能待独立设备步骤验证。`.2480` 若结束，预检会拒绝，原预检不得另换保底重投。

`read_result.py --run <worker-output> --runtime <frozen-runtime> --scope <worker-scope.json> --scope-sha256 <SHA> --accounting <independent-accounting.json> --accounting-sha256 <SHA> --gpu-step <编号>` 接受实际 `sacct` 七列绑定文件。只有唯一指定 GPU 终态才读原 worker 终态及两份现地 hipprof 窗，独立核实际 PID/PCI、两个原 kernel、三对分母、384 次完整验证、12 次复制、含复制的宿主墙钟和各臂显存峰值。保留失败账单与部分分母。该读取器给出组件计时候选状态，独立 post 未消费时保留 `promotion_allowed=false`。

普通 standby 的生产入口绑定八张卡。同节点的 7+1 独立保底和单卡试验尚无获准的生产入口或单卡独立清场证据。已验证的旧混合步骤曾在同一八卡租约内守七张卡，但原 `.2292` 的 holder gate 拒绝。`.2299` 和 `.2424` 指明已检查的可见性，完整交接仍无资格。此包只渲染零卡命令，未停止或取得 `.2480`、`.2559`，未执行目标或提交 GPU。
