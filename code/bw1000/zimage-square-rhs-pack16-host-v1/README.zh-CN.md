# O01 Pack16 单步 HOST 准备

本入口使用原 `zimage-square-rhs-pack16-v1` 的冻结 claimant、五件运行包和实际 .2662 零 GRES 依赖终态。依赖观测只证明包、Torch 版本元数据和 ELF；该步骤没有执行 GPU、加载 Torch 或实际运行 hipprof。HOST 入口保留之后真实 worker 的两个路由、三对六臂、384 次计量矩阵乘和 12 次复制。

## 本机验证

从实验仓库根目录运行 `python -B -X utf8 code/bw1000/zimage-square-rhs-pack16-host-v1/check_local.py <一个不存在的本机私有回执路径>`。脚本调用真实 `host.py dry-run`，验证已归档的原依赖回执和五件运行包，然后捕获唯一拟执行 argv。虚拟节点 `f99r9n99` 仅用于测试。调度查询和失败结果是本机边界测试；回执不证明实际目标已承接试验。

## 实际承接条件

当前 05:44 普通补位固定日历仍覆盖 f12r2n10。普通补位不读取本包意图作为节点保留。HOST 会拒绝这一路径，直到有根独立审查且被运行中的普通补位驱动实际采用的精确节点排除，并确认没有旧范围在途。测试中的虚拟节点仅捕获 argv。

由根协调者出具本次普通发布身份、独立新鲜保护收据和同节点清场。材料包括以下固定成员。

- `--repo` 包含冻结 Pack16、原 .2662 公共证据、审计器和 GLM accounting helper，`--runtime` 为已审五件运行包，`--original-dependencies` 为实际 .2662 的私有 `original-outcome.json`。
- `--scope` 与 `--scope-sha256` 绑定原请求、标签、发布 commit、parent、gfx936、Torch/HIP、PCI、boot ID、独立预检 step、预检 SHA、保护收据 SHA 及单个 240 秒 epoch。相同 scope 文件以只读方式进入 worker。
- `--pre` 是目标节点独立零 GRES 清场结果，含真实四 CPU 以上候选能力核查所需的 Slurm 归属、PCI 集合、boot ID SHA、零 KFD/设备 FD 和 1 DCU 物理承接证明。它与历史 .2662 的依赖检查用途不同。
- `--protection` 固定新鲜保护节点和步骤清单、所选节点及有效期。若现场选择 `f12r2n10`，还必须提供 `released_predecessor_step=1103606.2930`，并由 HOST 从原句柄独立读取实际终态。预估预算不构成前序终态；禁止 `f12r2n11`。HOST 在现有业务锁内校验归属、写入独占意图、启动及短时观察数值步骤，随后释放全局锁。运行及终态轮询在锁外；无接管、延长或取消其他步骤。
- `--target-runtime`、`--target-scope` 是只读源，`--target-output` 是独立且不存在的新目录。实际执行时路径必须与 HOST 读到的原件一致。实际 HOST 还需提供 --ordinary-calendar 与 --target-gate；前者必须是普通补位实际采用的已核日历身份，后者是随本包冻结的 worker_gate.py。计算节点先独立读取真实数值步骤及 TRES，成功后才进入冻结 DTK 和 runtime/site。上限 8192 MiB、4 CPU、1 DCU，HOST 为调度结果和回收留出同一 epoch 内的余量。

`dry-run` 允许注入观察时钟以复现 argv；`run` 与 `finish` 使用现场时钟。只有 `run` 可能创建输出并提交一个 GPU step。未获唯一终态时保留 UNKNOWN，禁重放。独立 post 在原 GPU 终态之后另行完成，再由 `finish` 核对独立 Slurm accounting、同 boot/PCI 集合、零设备进程及原冻结 worker 的路由、成对结果和完整分母。外层超时只回收本次拥有的 srun 客户端；不会以此宣告 GPU 已终态。任何失败保留计划分母，且不会把组件速度当作训练速度。
