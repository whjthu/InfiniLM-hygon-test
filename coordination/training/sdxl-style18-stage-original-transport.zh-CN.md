# SDXL style18 原数据请求的有限传输

将冻结的私有包 `train-only.zip` 从现有数据目录传向原目标范围。原请求 `274f1a27b7ea44f198eaa08589b845b49` 接收严格 290733 字节，SHA-256 为 `d860465dc1dde228a1fabbc716f581f2adbe67fe72f2f31841305023a9a0c302`。冻结目标端使用 `sys.stdin.buffer` 且上限 327680 字节。

根协调者先确认原 R1 认证和目标副本的已发布接收器。使用现有不分配 PTY 的 OpenSSH 参数前缀，把已冻结的远端命令作为最后一个参数交给 `subprocess.run`。下面的前缀由根原入口持有，工具不读取凭据和端点。

```python
import shlex
import subprocess

def transport(command, *, input, timeout):
    return subprocess.run([*approved_r1_argv, shlex.join(command)],
                          input=input, capture_output=True, timeout=timeout)
```

调用 `dispatch_sdxl_style18_stage_once.receive_once(verified_remote_checkout, transport)`。入口先发送空的二进制 stdin 做原 `inspect`。只有原目录确实缺失才独占创建本机 `local-only/training-sdxl-style18-train-staging-v1-original-dispatch/intent.json`，刷盘后将原 ZIP 字节交给一次 `receive`。接收端仍保留原 60 秒预算，本机一次调用上限为 75 秒。私有包不通过 PowerShell 文本管道或远端命令行参数传送。

若回复丢失、输出异常、原目录部分写入或本机已有意图，仅调用 `dispatch_sdxl_style18_stage_once.inspect(verified_remote_checkout, transport)` 查询原目录。该调用使用同一 SHA 和空二进制 stdin。成功的接收回复保留在本机，独立 `inspect` 要核对请求、包 SHA、数据绑定 SHA、13 个成员和 289527 字节，才给出目标数据已复核的结论。原正式 T06 数值失败保留，新域训练 READY 仍需来自独立的真实训练准入。
