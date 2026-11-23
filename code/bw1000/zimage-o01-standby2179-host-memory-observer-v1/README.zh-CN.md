# 原 .2179 保底步骤的宿主内存观察

新入口只读 .2179。申请同一节点独立 0 GRES、1 CPU、256 MiB、60 秒的步骤，使用已验证的线程绑定。上线前保持原工作进程 PID 982368、启动 ticks 356482943 和原私有 boot。不得取代或信号原持有者。

    python3 -I -S -B observer.py --source-sha256 SHA --attested-source STAGED/node_entry.py --boot-sha256 BOOT_SHA --previous-worker-pid 982368 --previous-worker-startticks 356482943 --out NEW_RESULT.json

入口按完整步骤号核对 .2179 和自身，复用已验证的 PID、启动 ticks、listpids 和 cgroup 归属函数。根据 .2179 进程的实际 memory controller 及观察者挂载表读取精确 step 层级。记录当前使用、上限、可读峰值、事件和分类，并列祖先中可见的硬上限。cgroup v1 failcnt 表示限额事件，只有支持的 OOM 字段才单列。结束时重验原进程、步骤及 boot。结果只包含小计数和身份摘要。

这是运行中 .2179 的宿主 RAM 观察。HBM 和原 .2175 已终结前的 cgroup 峰值仍未知。分类可能重叠，单张卡和任意时点使用量不能由该数据推算。目标只读结果仍待执行。

## 一次性协调端入口

从海光工作区调用 `launch_once.py --source-sha256 SHA --receipt-dir 新私有目录`。入口沿用已固定的认证配置，发送前核对集群原父作业、正在运行的八卡 .2179 和新的唯一观察标签。独占写入原观察范围后，调用一次 1 CPU、256 MiB、零卡的 Slurm 步骤，完成后核对独立账务和原保底工作状态。传输状态未知时，按原请求和结果路径对账。
