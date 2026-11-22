# T09 HunyuanVideo 1.5 第 10 步外部恢复

新请求 `644d698cb83f54b40fa8f897ffde427a` 在父作业 `1087656` 的 `f11r4n17` 上预留 1 张 DCU、8 CPU、128000 MiB。新的训练进程从原 `.794` 的 `checkpoint-10/state.pt` 恢复，执行固定缓存上的第 11–25 步。原检查点为 7,479,771 B，SHA256 为 `4ce48e5ea7f4cab5da1df7af908396b5dcf717c9ff2d8f3ba4cef14fe2217846`。使用原共同框架、Muon 优化器、模型和缓存身份。对照原 `.794` 第 11–25 步每步 loss 和 grad norm，以及第 25 步的元数据、优化器等可变状态、RNG、adapter 和完整性摘要。另记第 25 步归档字节是否相同。

原 `.828` 调度失败和 `.830` 对其五步恢复的独立通过分别保留。可选进程内自测的 JSON null 表示本次未执行该分支。实际新进程恢复必须另有起始步、执行步数、逐步记录和状态对照。

发布后由根协调者使用新鲜的 PCI、KFD、PID 和 Slurm 观测取得唯一准入，保护同节点原持有记录。将 `SOURCE_FILES.json` 及其所列文件精确安装到目标 `results/training-hunyuanvideo-cached-resume10-target-v1/src`。使用一次实际起始 epoch 贯穿下列各阶段。

```bash
python=/usr/local/python3.12/bin/python3
run=/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-cached-resume10-target-v1
"$python" -B "$run/src/preflight_ingress.py" --epoch "$epoch" --source-manifest-sha256 "$manifest_sha"
```

目标 HOST 准入读取已封原 `.828` preflight 和修正后的公开 `.830` 证据，核对当前文件、运行时、原 `.794` 终态和第 10 步检查点。预检成功后使用返回的 `dispatch.argv` 在同一 epoch 下唯一执行。控制器以两个单独的 0-GRES 步获取前后设备证据，核对带完整列的三阶段实际终态。预算为总计 1440 秒，工作阶段到 epoch + 1240 秒，GPU 步最多 1200 秒并受剩余时间约束。响应不明时对照原 intent 和编号步骤。

在原终态后由根独立核实会计及清场，再消费原记录。

```bash
"$python" -B "$run/src/read_terminal.py" --epoch "$epoch" --source-manifest-sha256 "$manifest_sha" \
  --preflight-sha256 "$preflight_sha" --output "$run/independent-terminal.json"
```

此范围验收同平台恢复一致性。A100 正式吞吐、留出集效果和 17 项正式训练验收仍需各自实测。
