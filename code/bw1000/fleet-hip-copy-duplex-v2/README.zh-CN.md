# O01 双向搬运先导 v2

原 v1 请求 773e6c71d59d885f3c8fa91602085be3 的实际 GPU 步骤 1103606.630 在 1 秒后 FAILED 1:0，获配 8 DCU，0/8 有效卡。初始化阶段的共享库文件名校验拒绝了 DTK canonical basename。v1 原请求、终态和清场单独保存。

v2 沿用 GLM 1103606.624 已验证的精确例外。在 worker 唯一 HIP 绑定中显式启用 exact_dtk_hip=True。例外同时要求 libgalaxyhip.so.5.2.26113.2065-6699a1f0 和 b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b，继续核绝对 canonical 路径、实际文件 SHA-256 和符号 dladdr provider。其他库和未显式启用的调用仍沿用原规则。

固定 O01 1103606/f12r2n13、12 CPU、12 GiB、8 DCU、150 秒 Slurm 及同一预算原始 epoch。沿用 1/32 MiB、serial/overlap/overlap/serial、每方向每组 8 次、每臂 3 组预热和 12 组计时。启动前沿用原 CPU-only fresh 准入和共享预约，终态后取得独立零卡清场。原 630 预约需要独立清场才能释放。

协调者使用本目录 launch.py::ship(publication,snapshot,existing_transport) 发送唯一新请求。其新安装目录下的 pilot.py --reconcile 只读取原件。完整测量定义见 [v1 说明](../fleet-hip-copy-duplex-v1/README.zh-CN.md)。精确 HIP 例外的真实组件证据见 [GLM 624](../../../results/bw1000/2026-11-22/GLM-I128-original624-independent-paired-component.json)。
