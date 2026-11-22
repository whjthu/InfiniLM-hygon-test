# GLM 首段 T512 的两节点 Slurm 工程验证 v4

本包把已冻结的 resident 首段 T512 调用器接到 I02 的同父两节点候选 1103606 / f13r1n08、f13r1n09。keeper 的静态候选为 f12r2n13。实际 GPU 请求是 1 个 Slurm step、每节点 8 DCU、64 CPU、224000 MiB。keeper 是 2 CPU、8192 MiB、0 GRES。pre 和 post 分别使用两个同组节点，各 1 CPU、1024 MiB、0 GRES。独立检查 43943 和 43944，点对点控制使用 43944。具体 IP 和当前占用由实查填入 claim，历史 map 仅支持候选选择。

同一个 launch epoch 管理所有预算。构造限 1800 秒，加载限 10800 秒，工作限 12120 秒，worker 完成限 12300 秒，GPU 截止 12480 秒，keeper 截止 12900 秒，总预算 13200 秒。GPU 申请前核对当前父作业、步骤、每节点八个 DRM PCI 与申请卡位、IP、端口、boot 和所有可读持有者。原零 GRES 诊断 1103606.1088 记录三台机器读取 KFD 逐卡拓扑均被 EPERM 拒绝，原 1103606.1087 失败后没有 keeper 或 GPU 派发。本包在零 GRES 原件中明确记为 UNAVAILABLE_ZERO_GRES。进入 GPU step 后，两台节点必须先在 HOST 上分别取得八张卡的完整实际 KFD 拓扑，与 DRM、claim 卡位和同 boot 对齐。两份验证均完成后才允许任一节点进入容器。任一节点权限拒绝、拓扑错配或超时均挡住模型入口。容器内每个 leader 和 owner 重核两份已分配的证据，leader 按 .1071 的实际 recipe 验证 71 个依赖路径和 SHA。

每个节点有一个 leader 和八个独立 owner。rank 0 调用真实 .1071 CABI 生成两个 UID，校验实际 step、source、epoch 及原 leader 记录的 producer PID/startticks。双向 ready 帧在两个已核 IPv4 间绑定至新请求，逐步验证 16 个 rank。每 owner 保持固定四 CPU、32K、原 checkpoint 的 117060 项及 78 层。完整 1312 ID 两臂均为 512、512、288，观察首段、三段 CABI 合计、宿主窗口、所有命中回退、rank 0 BF16 对照与 8 GiB 显存门槛。

## 安装和实际入口

在只读目标 stage 保留本目录所有文件，将 fresh-kv-prefill512-resident-pair-source-v1 完整复制为 pair/。下列只读依赖放到 stage 顶层并保持 contract.py 的冻结 SHA。

| stage 文件 | 来源 |
|---|---|
| ready_socket_parent.py、slurm_accounting.py、ENV.sh、checkpoint.py、INPUT_BINDING.json | 原 full-model-screen32-v1 |
| snapshot.py | 原 full-model-paired-v2，复用通用 DRM/KFD/FD 函数 |
| typed_binding.py | 实际 .1071 stage 的 generated/typed_binding.py |
| 1071.json | GLM474-t512-native-original1071-host-verified.json |

使用 contract.validate_claim 校验具体 claim。新请求为 e1549a5272f54350a567a927c9f10f39，名称为 GLM474-prefill512-resident-pair-GPU-v4。原 v3 的 GPU 1103606.1129 双节点读到完整 8 卡 KFD 拓扑，随后另一节点在首个可见但未写完的门禁 JSON 上失败，独立 post 已清场，0 个 owner 启动。v4 先将凭据写入同目录随机临时文件，经 flush、fsync、关闭后，以不覆盖原位的硬链接发布。更新凭据通过原子 replace 发布。发布失败会清理临时文件并保留已有凭据。节点门禁、leader 首次记录、UID、owner 进展及 keeper 凭据共用这个写入入口。填入实际检查的 I02 data_IPs、PCI_by_node 和 root fresh admission 字节 SHA，并记录 GPU_GO=true 的正式认领与当前资源互斥。PCI_by_node 要求 n08、n09 各含按字典排序的八个完整 DRM BDF。仓库本机 scope 尚无这些现场数据。

```sh
/usr/local/python3.12/bin/python3 -B host.py launch \
  --claim /absolute/approved-claim.json --claim-sha256 "$CLAIM_SHA" \
  --source-sha256 "$SOURCE_SHA" --admission /absolute/actual-admission.json
/usr/local/python3.12/bin/python3 -B host.py status
```

固定名称及排他的新目录保护原步只发一次。结果目录记录 intent、唯一 keeper PID、全部计费列、GPU/keeper/独立 post、两个设备门禁状态和失败分母。终态观测失败时保留预约。实际 GPU 门禁或模型失败且终态可见时仍执行零 GRES 独立清场，核对原 boot、精确 DRM PCI、原进程 PID/startticks 退出和端口空闲。每个节点有通过的门禁原件时，post 绑定该原件、原计费 step 和同一组 PCI，结果记录 prior_leased_KFD_step。未到门禁的节点将该字段记为 null。代码通过确切 Popen 句柄收回自身子进程。单序对照验证首段在完整模型的命中与数值，正式性能和服务资格仍待验证。

本机 check_local.py 使用真实并发线程及文件系统延迟写入，运行现有两节点门禁读取器。读取器在写入期间只观察到文件未发布，在发布后验证完整记录。测试覆盖重复目标原始字节保留、原子 replace、空和损坏 JSON 的严格读取，以及实际 launch/status CLI 的模拟 Slurm 结果。目标共享文件系统的硬链接能力仍待验证，链接失败会停止发布。
