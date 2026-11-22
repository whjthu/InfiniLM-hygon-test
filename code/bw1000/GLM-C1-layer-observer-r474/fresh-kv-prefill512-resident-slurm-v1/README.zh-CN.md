# GLM 首段 T512 的两节点 Slurm 工程验证

本包把已冻结的 resident 首段 T512 调用器接到 I02 的同父两节点候选 1103606 / f13r1n08、f13r1n09。keeper 的静态候选为 f12r2n13。实际 GPU 请求是 1 个 Slurm step、每节点 8 DCU、64 CPU、224000 MiB。keeper 是 2 CPU、8192 MiB、0 GRES。pre 和 post 分别使用两个同组节点，各 1 CPU、1024 MiB、0 GRES。独立检查 43943 和 43944，点对点控制使用 43944。具体 IP 和当前占用由实查填入 claim，历史 map 仅支持候选选择。

同一个 launch epoch 管理所有预算。构造限 1800 秒，加载限 10800 秒，工作限 12120 秒，worker 完成限 12300 秒，GPU 截止 12480 秒，keeper 截止 12900 秒，总预算 13200 秒。GPU 申请前检查当前父作业节点、已知步骤，以及一次新鲜的硬件、进程和端口状态。每个计算节点在 GPU step 内再次核验同 boot、8 个 PCI/KFD、数据 IP 和端口。容器启动后按 .1071 实际 recipe 验证 71 个依赖的路径与 SHA。

每个节点有一个 leader 和八个独立 owner。rank 0 调用真实 .1071 CABI 生成两个 UID，校验实际 step、source、epoch 及原 leader 记录的 producer PID/startticks。双向 ready 帧在两个已核 IPv4 间绑定至新请求，逐步验证 16 个 rank。每 owner 保持固定四 CPU、32K、原 checkpoint 的 117060 项及 78 层。完整 1312 ID 两臂均为 512、512、288，观察首段、三段 CABI 合计、宿主窗口、所有命中回退、rank 0 BF16 对照与 8 GiB 显存门槛。

## 安装和实际入口

在只读目标 stage 保留本目录所有文件，将 fresh-kv-prefill512-resident-pair-source-v1 完整复制为 pair/。下列只读依赖放到 stage 顶层并保持 contract.py 的冻结 SHA。

| stage 文件 | 来源 |
|---|---|
| ready_socket_parent.py、slurm_accounting.py、ENV.sh、checkpoint.py、INPUT_BINDING.json | 原 full-model-screen32-v1 |
| snapshot.py | 原 full-model-paired-v2，复用通用 DRM/KFD/FD 函数 |
| typed_binding.py | 实际 .1071 stage 的 generated/typed_binding.py |
| 1071.json | GLM474-t512-native-original1071-host-verified.json |

使用 contract.validate_claim 校验具体 claim。填入实际检查的 I02 data_IPs 和 root fresh admission 字节 SHA，并记录 GPU_GO=true 的正式认领与当前资源互斥。仓库本机 scope 尚无这些现场数据。

```sh
/usr/local/python3.12/bin/python3 -B host.py launch \
  --claim /absolute/approved-claim.json --claim-sha256 "$CLAIM_SHA" \
  --source-sha256 "$SOURCE_SHA" --admission /absolute/actual-admission.json
/usr/local/python3.12/bin/python3 -B host.py status
```

固定名称及排他的新目录保护原步只发一次。结果目录记录 intent、唯一 keeper PID、全部计费列、GPU/keeper/独立 post、有限观察和失败分母。终态观测失败时保留预约。真实 GPU 失败且终态可见时仍独立清场，核对同 boot/PCI/KFD、原进程 PID/startticks 退出和端口空闲。代码通过确切 Popen 句柄收回自身子进程。单序对照验证首段在完整模型的命中与数值，正式性能和服务资格仍待验证。

本机 check_local.py 覆盖两端真实 socketpair 帧、原 helper 字节、资源命令、记账重复、GPU 失败与 UNKNOWN、陈旧硬件及 PID/步骤身份。目标调用数为零。
