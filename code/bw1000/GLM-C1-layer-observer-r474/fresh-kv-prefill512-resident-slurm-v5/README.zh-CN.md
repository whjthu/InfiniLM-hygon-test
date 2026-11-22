# GLM 首段 T512 的两节点 Slurm 工程验证 v5

本包把已冻结的 resident 首段 T512 调用器接到 I02 的同父两节点候选 1103606 / f13r1n08、f13r1n09。keeper 的静态候选为 f12r2n13。实际 GPU 请求是 1 个 Slurm step、每节点 8 DCU、64 CPU、224000 MiB。keeper 是 2 CPU、8192 MiB、0 GRES。pre 和 post 分别使用两个同组节点，各 1 CPU、1024 MiB、0 GRES。独立检查 43943 和 43944，点对点控制使用 43944。具体 IP 和当前占用由实查填入 claim，历史 map 仅支持候选选择。

同一个 launch epoch 管理所有预算。构造限 1800 秒，加载限 10800 秒，工作限 12120 秒，worker 完成限 12300 秒，GPU 截止 12480 秒，keeper 截止 12900 秒，总预算 13200 秒。GPU 申请前核对当前父作业、步骤、每节点八个 DRM PCI 与申请卡位、IP、端口、boot 和所有可读持有者。原零 GRES 诊断 1103606.1088 记录三台机器读取 KFD 逐卡拓扑均被 EPERM 拒绝，原 1103606.1087 失败后没有 keeper 或 GPU 派发。本包在零 GRES 原件中明确记为 UNAVAILABLE_ZERO_GRES。进入 GPU step 后，两台节点必须先在 HOST 上分别取得八张卡的完整实际 KFD 拓扑，与 DRM、claim 卡位和同 boot 对齐。两份验证均完成后才允许任一节点进入容器。任一节点权限拒绝、拓扑错配或超时均挡住模型入口。容器内每个 leader 和 owner 重核两份已分配的证据，leader 按 .1071 的实际 recipe 验证 71 个依赖路径和 SHA。

每个节点有一个 leader 和八个独立 owner。rank 0 调用真实 .1071 CABI 生成两个 UID，校验实际 step、source、epoch 及原 leader 记录的 producer PID/startticks。双向 ready 帧在两个已核 IPv4 间绑定至新请求，逐步验证 16 个 rank。每 owner 保持固定四 CPU、32K、原 checkpoint 的 117060 项及 78 层。完整 1312 ID 两臂均为 512、512、288，观察首段、三段 CABI 合计、宿主窗口、所有命中回退、rank 0 BF16 对照与 8 GiB 显存门槛。

## 安装和实际入口

在只读目标 stage 保留本目录所有文件，将 fresh-kv-prefill512-resident-pair-source-v2 完整复制为 pair/。下列只读依赖放到 stage 顶层并保持 contract.py 的冻结 SHA。

| stage 文件 | 来源 |
|---|---|
| ready_socket_parent.py、slurm_accounting.py、ENV.sh、checkpoint.py、INPUT_BINDING.json | 原 full-model-screen32-v1 |
| snapshot.py | 原 full-model-paired-v2，复用通用 DRM/KFD/FD 函数 |
| typed_binding.py | 实际 .1071 stage 的 generated/typed_binding.py |
| 1071.json | GLM474-t512-native-original1071-host-verified.json |

使用 contract.validate_claim 校验具体 claim。新请求为 7649a03ee7544ca3ab255cb82e555473，名称为 GLM474-prefill512-resident-pair-GPU-v5。原 1103606.1143 已完成两台各 8 卡的 KFD 门禁并写出 16 份 owner 记录。入口读取 config 时，原有 16384 字节上限拒绝了实际 29464 字节文件，0 次前向，原 post 与独立清场已通过。v5 以原 SHA 冻结 config 的 29464 字节和 index 的 11359251 字节，并保留 v4 经现场验证的原子凭据发布。填入实际检查的 I02 data_IPs、PCI_by_node 和 root fresh admission 字节 SHA，请求正式资源时保留互斥。PCI_by_node 要求 n08、n09 各含按字典排序的八个完整 DRM BDF。

独立预检可先在同父 n08、n09 的双节点零 GRES 步骤运行。把 evidence 目录放在项目 results 下，预先创建并使用新名称。容器挂载与正式设备步骤一致。

```sh
srun --jobid=1103606 --job-name=GLM474-T512-v5-mounted-admission \
  --exact --immediate=8 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --nodelist=f13r1n08,f13r1n09 --cpus-per-task=1 --mem=1024M \
  --gres=none --time=1 --kill-on-bad-exit=1 \
  /bin/bash "$STAGE/container.sh" --mounted-preflight "$EVIDENCE_DIR" "$SOURCE_SHA"
```

每个节点输出 mounted-pre-节点.json，只含九件原件的字节数、SHA 和固定形状。正式 keeper 的 pre 步骤同时运行此入口，核对其真实零 GRES 记账及两个 exact 挂载凭据才继续派发 GPU。检查读取原 config、index、私有输入计划、公有输入绑定、checkpoint loader、typed binding、.1071 编译证据、ELF 文件和 recipe。任务的数据留在集群，预检无需权重字节、Native 加载或 SDK 调用。

```sh
/usr/local/python3.12/bin/python3 -B host.py launch \
  --claim /absolute/approved-claim.json --claim-sha256 "$CLAIM_SHA" \
  --source-sha256 "$SOURCE_SHA" --admission /absolute/actual-admission.json
/usr/local/python3.12/bin/python3 -B host.py status
```

固定名称及排他的新目录保护原步只发一次。结果目录记录 intent、唯一 keeper PID、全部计费列、GPU/keeper/独立 post、两个设备门禁状态和失败分母。终态观测失败时保留预约。实际 GPU 门禁或模型失败且终态可见时仍执行零 GRES 独立清场，核对原 boot、精确 DRM PCI、原进程 PID/startticks 退出和端口空闲。每个节点有通过的门禁原件时，post 绑定该原件、原计费 step 和同一组 PCI，结果记录 prior_leased_KFD_step。未到门禁的节点将该字段记为 null。代码通过确切 Popen 句柄收回自身子进程。单序对照验证首段在完整模型的命中与数值，正式性能和服务资格仍待验证。

本机 check_local.py 覆盖预检生成与两份原始凭据消费，缺一份或收到旧大小都会阻止 GPU。既有回归还覆盖真实并发文件发布、重复目标、严格 JSON 和实际 launch/status CLI 的模拟 Slurm 结果。真实挂载预检留待集群验证。
