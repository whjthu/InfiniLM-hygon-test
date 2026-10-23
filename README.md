# InfiniLM 海光协同优化实验仓库

用于 BW1000 / BW1100 两地 agent 交换代码、交接材料、性能数据与失败结论。只使用普通 Git，不依赖 GitHub API。

先读 [协作流程](coordination/PROTOCOL.zh-CN.md)、[Agent 约定](AGENTS.md)；BW1100 接入使用 [这份任务书](handoff/BW1100_AGENT_START.zh-CN.md)。

## 当前入口

- [BW1000 状态](status/bw1000/latest.json)：包含观测时间与证据，不把历史运行状态当作当前进程存活证明。
- [BW1000 当前性能与瓶颈](analysis/bw1000/current-state.zh-CN.md)。
- [结果目录](results/bw1000/2026-10-24/)：模型、组件、失败记录分开。
- [近期实验源码](code/bw1000/)：逐文件保留原字节与 SHA。部分实验依赖旧私有父版本；并非都已是可独立构建的产品代码。
- [此前已发布的交接源码归档](archives/bw1000/)：保留旧双版本、依赖及许可证，避免丢失基线。
- [待认领任务](coordination/tasks/)。

这里是实验协作空间。最终交付另开 `delivery` 分支整理代码、必要测试和运行说明，移除实验交接/大批性能数据目录；实验历史仍保留在本仓库，不改写历史。

验收文件完整性：`python tools/verify_repository.py`。此命令不执行实验代码、不访问 GPU/服务器。
