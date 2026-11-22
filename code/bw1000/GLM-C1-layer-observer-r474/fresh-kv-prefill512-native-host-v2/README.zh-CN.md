# GLM 首段 T512 Native HOST 构建 v2

原 1103606.1068 目标 HOST 步在编译前失败。旧引导将 PurePosixPath 交给了 .855 终态验证器，后者需要在实际 compute 挂载上打开文件。新范围在这一 I/O 边界传入具体 Path，保留 POSIX 形式的配方路径。原请求和结果目录继续作为独立失败证据。

这一范围从已发布的首段 T512 caller 渲染精确源码。它只编译 u00、u15、u17 三个 HOST 翻译单元，复用已验证的 .855 设备对象和 .918 配方中的另外 29 个对象。目标先核对实际 .918 BUILD_RECIPE 字段及原 DSO SHA，再做三份 fresh M。每份实际 MD 与所属 fresh M 一致后才链接。

目标采用 f12r2n13，父作业 1103606，4 CPU、16 GiB、零 GRES。HOST 冻结唯一 epoch，work_deadline 为 epoch 加 1680 秒，deadline 为 epoch 加 1800 秒。对同会话的重复检查用 reconcile。目标链接必须继续闭合 71 个原 provider，并且为 .918 已验证的 284 个强导出只添加 modern_glm_global_fresh_kv_prefill512_r474。检查读取也保留旧符号和 NEEDED 证据。

```text
python -I -S -B prepare_stage.py <new-checkout>
python -I -S -B stage_packets.py <checkout> <new-packet-directory> --claim-sha256 <claim> --stage-manifest-sha256 <stage-manifest>
python -I -S -B check_local.py -v
```

根协调者校验普通发布和新鲜设备状态后安装精确 40 个成员。HOST 入口为 host_once.py，传 --publication、--claim-sha256 和 --stage-manifest-sha256，再选 --preflight、--launch 或 --reconcile。实际 .1068 失败、.918/.919/.920、.855/.856/.857、CPU308 回执及当前 Slurm 路由还需现场一致。若登录挂载看不到编译产物，终态保持未验证，使用独立零卡 compute 读取。

```text
worker.py --audit-compute <claim-sha> --intent <original intent> --accounting <fixed seven-column accounting> --kernel-reconciliation <original proof>
```

本机验证覆盖规范请求、源包、有限传输接收、旧对象有序沿用、MD 不一致阻止链接和符号范围。尚未安装目标包或在目标环境编译。此范围仅用于 HOST 构建，保持 GPU 和 Model 准入关闭。
