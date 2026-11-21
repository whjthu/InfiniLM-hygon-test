# DeepSeek-v4 C4 带权对照的运行时归属接线

本入口把 `1103606.811` 的探针和 `1103606.487` 的候选 Native 接入原 13 C ABI 加载器。源与执行范围由另一次设备认领绑定。既有 `.811` 只完成被动 ELF 核验，`.812` 是其独立清场。

未来固定的设备包将原 `.156` 与 `.811` 公开原件分别封入 `original-156-terminal.json`、`original-811-terminal.json`，并复用原 `BUILD_PLAN.json`、`check_runtime_owner.py`、`CHECKPOINT_METADATA.json`、`entry.py` 和 `range_receipt.py`。本入口在容器内读取 `/refactor/candidates/DeepC4-T1-history-guarded-probe-link-v1/CPU-result.json` 和同目录 `link-plan.json` 的原始字节，复核 69 项依赖闭包和唯一本地 Native 替换。之后使用原校验器检查实际加载的六个项目提供方、全部 13 个入口的 `dladdr` 归属，以及打开前后的文件身份。

根编排器先通过原单卡控制器核对新鲜 PCI、Slurm 步和独立进程身份。设备子进程再调用 `bind_after_admission(package, Path("/refactor"), arm, original_receipt_sha, deadline)`，`arm` 为 `baseline` 或 `candidate`。它先校验原七段回执，然后返回 `probe, binding, ownership`，调用方保存 `ownership` 及真实设备、步骤和进程证据后将 `probe, binding` 交给原 `run_weighted`。两臂须使用同一物理卡上的独立进程、原权重回执和 32768 因果输入。发布前将同卡和不同进程证据与原比对器的 71 步、输入护栏、六组独立数值采样、缓存审计和三组输出绑定。

原检查请求为 `7ae20a906e2646b1bbcea3e712cb6bba`。最新仅确认 `/refactor/results/Deep483-layer2-C4-indexer-original-seven-weight-range-pins-v1.json` 未观察到，原执行状态仍未知。当前无需重复原检查器或已完成的 `.159/.162/.811`。新入口的本机验证使用两份公开原件、原依赖清单，以及隔离的链接回执和加载替身。未执行实际 dlopen、权重读取、GPU 或 Native 分支观测。
