# T12 双向训练视图：25 次连续更新和 5 次新进程恢复

这是一个新数据轴训练任务。输入只用已在海光运行过的 near、nostructure_texture_far 两个训练场景及原 17 文件，不下载、不增加场景，也不接触 Cabinet/Teddy 开发集或 household。两对原始 RGB-D 已分别做过正反向重投影门禁。每对新增反向视图时，图像、深度、内参、掩码同步换序；OpenCV world-to-camera 姿态重新以新首相机为坐标原点，得到 `I, inverse(E1)`。它是同媒体的明确派生增强，不作为新独立样本或泛化证据。

统一 a68952cb/393 文件框架的模型、相机头、BF16、AdamW、固定 2e-5 学习率、权重衰减 0.01、裁剪 1.0、seed 20260911 和原 camera loss + 0.1 direction loss 保持原配方。仅数据顺序变为 near 正、planar 正、near 反、planar 反。现有框架的 view_order 和 tensor-content 身份包含全部四视图，游标和断点原生按 4 取模；不能继续使用旧两 batch 的游标检查。

一次 scope 预注册两个独立 Python 进程：continuous 完成 25 次真实 optimizer 更新（5 步预热 + 20 步计时），保存 10/20/25 断点；resume20 从同 scope 的原第 20 步完整断点恢复，再做 5 次更新。总计 30 次调用，不是两轮性能测试。恢复比较保留 loss/grad/参数、完整 AdamW、placements、RNG、完整边界 SHA 和实际输入游标，阈值沿用原 1e-6 loss / 完整状态 SHA。无独立 scheduler 对象，固定 LR 存在 optimizer param_groups。

数据来源遵守已有 TUM RGB-D 来源/许可记录和作者署名。图像与标签留目标端；原数据 SHA 不变。新任务不复用或重放 `c55668b8ef104a9cb18be34d94251351` 未知八对请求，也不复测已完成的 directed45 或 lowLR45。它们只提供资产和成功执行依据。该任务不宣称正式 CO3D、领域效果或性能验收。

`launch.main(publication, transport)` 接受 root 已有普通 SSH 回调 `transport(body, stdout, stderr, timeout)`；回调必须按给定余量执行，并收口其原 SSH 子进程。整条本地准备/Git/传输/HOST/Slurm/训练/独立 post 共用开始时捕获的 1800 秒 epoch。GPU 上限 1440、容器 1300；Slurm 实际期限和客户端等待按剩余时间收缩。HOST controller 在既有 SSH 内前台运行，不创建后台服务。UNKNOWN 不重提。所有 GO 初始 false；root 审核后仅采纳这次训练执行权限，Model/formal17/effect/performance 仍 false。

建议目标是已实测同资产的 parent 1103606 / f13r4n09、1 DCU、4 CPU、64 GiB，实际必须先通过当前父作业成员/同 UID、无活动 step、PCI/KFD/PID/端口和模型身份门禁。原479的 45 次更新约375秒只是旧工作负载实测，不是新任务时延或显存上界。新进程恢复的开销、训练输入 hash、实测显存和实际计数都须由目标执行记录；CPU 几何契约不构成目标训练资格。

当前只有本地源码和合成几何边界检查；没有新目标安装、Slurm、GPU、Torch 导入、权重读取或 optimizer 更新。root 需审阅、普通发布、唯一派发，并独立消费真实终态后才能把本任务记为完成。效果评估另立 scope，保留尚未测量状态，不以 loss 降低替代任务指标。
