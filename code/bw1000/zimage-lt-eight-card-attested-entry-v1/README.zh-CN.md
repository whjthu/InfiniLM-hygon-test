# Z-Image 八卡监督器的 Slurm 归属检查

在已获准的八卡步骤中，宿主入口按完整步骤号读取 Slurm，确认节点、步骤标签、12 CPU、8 DCU 和任务 PID。写入包含 PID、启动时刻、CPU 亲和性、boot 与 cgroup 哈希的私有收据，再进入固定 DTK 容器。容器重新核对同一范围与宿主 PID 身份，保存自己的收据后调用已冻结的监督器。两份收据均采用独占创建。整个入口沿用原 240 秒，剩余不足 185 秒时退出。

```text
python3 -I -S -B node_entry.py host --run RUN --runtime RUNTIME \
  --scope-sha256 SHA256 --entry-sha256 SHA256
```

`RUN` 必须包含协调者已绑定的 `scope.json`。`RUNTIME` 使用已发布的监督器运行清单。程序从宿主绑定已核验的 `/usr/local/hyhal`、原算子库、DTK 和 Python 依赖。PID 查询在目标计算节点上执行。 [Slurm scontrol 文档](https://slurm.schedmd.com/scontrol.html)

`probe` 是一次独立的零 DCU 检查，实际验证同一归属代码中的 CPU affinity、Slurm `listpids` 及 cgroup 指纹。

```text
python3 -I -S -B node_entry.py probe --run RUN
```

确定下一次 GPU 步骤前，仍需收到原保底的实际终态、零卡独立清场、共享锁内的唯一派发，及结束后八卡保底接续的证据。这里的入口只处理已分配的步骤。零卡实测的 PID 与 cgroup 证据不能代替八卡步骤的设备归属或算子结果。
