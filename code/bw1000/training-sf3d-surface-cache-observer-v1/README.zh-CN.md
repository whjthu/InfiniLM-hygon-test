# SF3D surface：原首训内的真实缓存观测增量

沿 root 已发布 `37d121c9` 的 train2/dev1 surface 路由继续，不重算标签、不重复 CLI/数据资格。此包在同一个原训练调用内补逐阶段私有进度及第一次更新前的真实缓存检查；**没有新增 cache-only 作业或训练循环**。当前只是源码与本机元数据边界检查，真实模型/缓存/训练均未执行，READY新增0。

## 可执行接入

`render_observed.render()` 返回6件共同框架覆盖层和 AST 复用证明：

- 复用原 CLI、atlas backend/loader、surface loader。
- 在原 surface backend 的设备检查、模型载入/迁移、每图 prepare_batch、scene_codes、query_triplane、image_features、material_fields 处包裹观测上下文。
- 新增 `generation/src/gencuda/backends/surface_cache_observer.py`，只写本次输出目录的 `surface-cache-observation/progress.json`。

renderer 先核 root 的 `render_native.py` 精确 SHA `d647811cfab3d835ad28884cf8c306b699806b9d834fefe59cb38d9fb006560d`，再核生成的原 surface backend 58679B / `c81d32e0103dc7cee323bd434af099cb7267273501dc8334530c621f0dc5300b`。剥离全部新增观测后，原函数 AST 必须完全相同，包括随机点生成、模型/损失、优化器、原训练与恢复循环。原 formal SF3D backend 不改。生成文件全部过 Python3.10语法检查。

根后续在独立目标目录安装共同框架及此6件 overlay 后，仍用原 `train stable-fast-3d` 和已经冻结的 `sf3d_surface` 私有配置；没有另起通道、调用模型的本机 launcher 或默认远端 fallback。覆盖层身份和 helper 必须纳入根的新源码清单/claim，不能把本包当成已经安装。

## 真正缓存检查，当前未执行

在原缓存生成/恢复之后、第一次真实更新之前：

1. 直接复用原 `_sf3d_capture_cached_batches` 将**真实**缓存转为 CPU，以及原 `_sf3d_validate_cached_batches` 对真实 CPU Tensor 校验；不得造 feature 张量。
2. 使用 live model 得到的 feature spec，额外检查实际 triplane 行数等于真实 `query_points`，两对象完整。
3. 检查缓存中 albedo/roughness/metallic 与此次已 SHA 绑定、按顺序加载的 surface 目标逐值一致，防止实际冷缓存误落回旧 RGB heuristic/错序标签。
4. 私有进度仅写每字段 shape/dtype/bytes/SHA、载入标签/输入 SHA、阶段及时长，不写 tensor 值；没有额外导出原权重、activation 或缓存。

这些真实 Tensor 调用本机 **UNRUN**，不是本机缓存资格。新增 CPU clone/hash/检查成本发生于首更新前，单列 validate_cache 阶段；原 measured training loop 不变，也不把此观测路径作为正式性能收益。原恢复/原缓存身份检查继续生效，不被 observer 替代。

## 失败与新本机检查

每次 `phase` 进入前原子保存 entered/completed=false，成功后保存 completed；失败保存阶段、对象序号和异常类型/消息前4096字符的 SHA，原异常继续抛出。诊断磁盘错误或异常字符串本身失败均不能盖掉原异常。进度上限64KiB，已有 observer 子目录即拒绝，不能借重开目录洗掉 UNKNOWN。原 Slurm/owner/预算/清理/post 沿根原启动器。

`check_CPU_once.py` 首次新7项检查已通过：生成6件源语法、原子进度、失败保留、目录独占、阶段/对象拒绝、64KiB上限、诊断写失败不遮原异常。另一次针对性检查确认异常 `__str__` 失败也保原异常，并闭合最终 renderer import 前 SHA 门禁；未重复原7项或旧 surface 数据/CLI 测试。仅 metadata fixture，**没有 mock feature Tensor、Torch、真实模型或缓存**。

## 仍然真实阻塞的准入

`coordination/tasks/training-t11-sf3d-original-weights-v1.json` 仍记录原两文件 checkpoint `COPY_NOT_EXECUTED_AUTO_REVIEW_APPROVAL_PENDING`：config.yaml 2405B / `5e880cfec3dec28d4e20afe07bfa1799c8a561f56f0532ebf1b4022ae4ddc8e4`，model.safetensors 4024289892B / `a3416e1cf654e7d4f5e75f116cec2c3f0a14501a77d30c2f6068bbda178de388`。本包只引用公开身份，**没有读取、转移、重提或下载这两个文件**，不绕审批。

既有 T11 CPU `.93` 只证明当时的 native imports/operator；不证明当前模型已载入、HIP forward、特征或更新。最短合规链仍是：根闭原 checkpoint 审批/实际身份和原意图→核当前私有 runtime/共同框架及真实 train2→fresh GPU/PCI/KFD/端口/原始时限→**同一原首训**产生真实缓存并过上述检查→真实 optimizer 更新→原终态及独立 post。此增量不为它们增加一项新 GPU 门，不计训练 READY。

## 数据与发布边界

数据仍为原公开 Khronos WaterBottle/Corset train、BoomBox dev，来源和 CC0/文档CC BY4 沿 root surface cohort 证据；未读开发/盲标签、图片或任何新数据原件。新 source 不改变既有分组/标签/采样和 formal17/A100参考。

只发布自编3份源码、说明及小证据，由根串行处理；进度、真实缓存和所有数据仍留私有目标输出目录。实际网络/SSH/Slurm/GPU/Torch/model/optimizer=0。
