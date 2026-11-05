# r470 唯一设备步骤原件审计

`910371.870` FAILED1/151s，keeper `910407.1065` FAILED1/153s；独立 failure-post `910371.871` COMPLETED0，两节点 KFD、已知 PID、记录的 UID listener、43843/43844 均清理。

16 个 created 标记证明原 create/geometry/floor/mask 检查路径到达；只有 rank0..7 在进入 combined proof 前持久化了 geometry。0 proved、0 normal close。worker0 在原150s deadline 后实际 TERM，worker1 的 finally 未完整落盘。不能推断 SUM136/order 正确、正常销毁、TP16 Model、缓存 fit 或性能，也不能唯一归咎 CCL/网络。没有新运行或自动重试。

从仓库调用（Python stdlib，不 import 生产入口）：

```
python -B code/bw1000/GLM952-global-TP16-component-r470/terminal-device870/read_original.py /absolute/path/GLM470-device870-terminal-sealed-v1.json.gz --output /absolute/path/audit.json
```

Reader 核全部74原件的逐字节 SHA/长度、原 sourceMF/claim、16 owner 身份/双UID哈希/同库maps/tuplePCI/阶段标记、原 accounting/forcedTERM 与独立 post。另从已发布 device-v1/GPU_MANIFEST.json 读取冻结元数据（其 SHA b0ea…，不是伪装进原归档）。`audit_passed=true` 表示失败原件自洽，`experiment_passed=false` 保留设备失败。原始 UID/完整 maps 留 R/L，不入 Git。
