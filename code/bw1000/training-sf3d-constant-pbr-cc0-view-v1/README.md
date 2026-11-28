# SF3D 固定视角恒定材质样本

将新的 GGX 着色施加到已核验的 Poly Haven 工具柜几何和固定相机，保留原可见面及 alpha。模型训练目标采用原材质头输入的物体级三个标签形状。独立回读核对完整 RGBA、97 个标量着色取点及 FP32 标签，分别扰动三个材质字段确认它们会影响实际成像。 [原始资产](https://polyhaven.com/a/metal_tool_chest) · [CPU 回执](../../../results/bw1000/2026-11-28/specialized-sf3d-cc0-constant-pbr-cpu-render-and-independent-readback.json)

`render.py` 和 `lineage.py` 是原执行文件的逐字节副本。复现时把它们分别放入同一检出目录下的 `local-only/sf3d-cc0-constant-material-fixed-view-v1/` 和 `local-only/sf3d-metal-tool-chest-pixel-lineage-v1/`。渲染器需要前序工具柜的资产、已校验固定视角和 `RESULT/surface-pixel-lineage.private.npz`。先在新的私有输出目录执行 `render.py build`，再执行 `render.py verify`。已存在的样本保留原件。

当前原生读取器认定的是历史 WaterBottle/Corset 样本组。要纳入统一框架的领域训练，请绑定新样本组，先实际读回原生冻结特征，核可训练的原 36 个张量，再记录真实优化器更新。当前结果覆盖样本制备与 CPU 核验。 [原生接口核查](../../../results/bw1000/2026-11-28/specialized-sf3d-metal-tool-chest-point-contract-and-head-compatibility.json)
