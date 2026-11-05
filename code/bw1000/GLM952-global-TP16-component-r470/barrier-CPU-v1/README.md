# r470 CPU-only marker barrier

一次两节点各8个独立 Python 子进程，直接 SHA 核验并调用已发布 device-v1/shared.py 的原 `save/wait`。全部16个新 created marker 等待同一固定序列；原旧目录只读且不修改。150秒同一绝对epoch、180秒Slurm上限，每节点1CPU/1GiB，无GPU资源。

使用相同外层容器/aliases、内部Python，只导入stdlib，没有SDK/Runtime/constructor/权重/Model。原 wait 前/后均持久阶段与耗时；静态无确定路径/UID/epoch/rank分叉。原870记录不足以判断NFS/CCL，当前PASS也不能追认历史状态。

发布后唯一 HOST 调用 `barrier.py launch --publication COMMIT`；xintent/原label sacct/目标存在即拒重放。未知时仅核原intent/PID/startticks/accounting，不重提。子进程失败只终止自己的Popen对象；独立CPU步骤正常终结后核全部16report/2worker/marker，不运行设备。
