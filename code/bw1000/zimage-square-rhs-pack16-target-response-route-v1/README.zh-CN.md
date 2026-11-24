# Pack16 已归档 .2662 目标回执入口

本入口只替换原 HOST 的历史依赖读取方式。保持原 `--original-dependencies` 调用时，仍由原 HOST 核验真实私有传输 envelope。采用替代入口时，不读取、构造或传输该私有文件。

替代入口要求原 .2662 目标 `launch-and-accounting.json` 保持在历史共享主机路径，且精确为 3,049 字节，SHA-256 为 d12faee2177a9e7e82076e333b1bba03ce55151381413b2d2d9767b035bf2edf。它还核对发布证明、原公开依赖结果、原响应编码器和第九轮独立目标归档。检查包括同一 request、完整九列 .2662 账单、f12r2n10、零 GRES、原 step_result 逐项相等、控制者记录的原传输确认及原 SHA。历史依赖不授权新设备步骤。

在正式已安装的 HOST 闭包上调用本入口。第一个参数保持原 HOST 模式 `dry-run`、`run` 或 `finish`，原 `--repo` 指向已重读的第九轮源码根。替代模式额外提供以下参数，且省去原 `--original-dependencies`。

- `--target-response` 为原共享主机 `results/O01-pack16-deps-json-c0243062d1f2/launch-and-accounting.json` 的完整路径。
- `--historical-proof` 和 `--historical-proof-sha256` 指向已发布 `coordination/tasks/O01-pack16-2662-target-response-identity-proof-v1.json`，SHA-256 为 3726ac745f9d10a45e718e60edc25ca438d28acf9094b41535b4f0d3f5e25fcb。
- `--round9-archive` 指向 `results/bw1000/2026-11-25/training-astra-sol-round9-pack16-original-shared-host-install-and-readback.json`。
- `--publisher-encoder` 指向原 `code/bw1000/zimage-square-rhs-pack16-live-preflight-v3/dispatch_host.py`。本入口只从该固定源读取编码函数。

本机检查使用原封存 envelope 和确切编码器重现目标 SHA，再对两条路径分别进入实际 Pack16 HOST dry-run。两者的冻结源码、运行时、审计器及生成的 argv 均一致。替代模式不会改写原普通仲裁、独占输出、真实设备资源、终态或独立 post 的要求。此包未在目标调用，未产生 Slurm 或 GPU 操作。
