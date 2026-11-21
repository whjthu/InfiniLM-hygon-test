# TRELLIS 三个公开对象的几何参考与评分输入

本包已实际取得 WaterBottle、Corset、BoomBox 的官方 glTF 引用网格 buffer，完成三次原生 `corner_geometry` 调用与参考点输入。它补足可执行的几何评估前置；真实模型预测、Shape Flow 缓存、训练、恢复和目标端资格均为零，不增加 READY。

| 来源对象 | 固定用途 | 顶点 | 三角面 | 固定面积采样点 |
|---|---|---:|---:|---:|
| WaterBottle | train | 2549 | 4510 | 2048 |
| Corset | train | 11505 | 18324 | 2048 |
| BoomBox | dev | 3575 | 6036 | 2048 |

同对象的图像、材质、网格、视角和采样点共用来源组，保持先前实际 train2/dev1/blind0 划分。6144 点不是 6144 个独立对象，也不是训练 token。跨组文件及数组精确 hash 无重复；未证明语义近重复排除或新地域/新作者泛化。只有一个开发对象，没有未暴露盲组。

## 来源与许可

发布方为 KhronosGroup/glTF-Sample-Assets。三个模型各自的 LICENSE 声明模型素材 CC0；许可证及仓库说明文档使用 CC BY 4.0。保留原作者的署名/权利说明、源文件身份和修改声明，不扩大到标识/商标授权。

- [WaterBottle 原权利声明](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/WaterBottle/LICENSE.md)
- [Corset 原权利声明](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Corset/LICENSE.md)
- [BoomBox 原权利声明](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/BoomBox/LICENSE.md)
- [glTF 2.0 官方坐标、accessor 与 node 变换约定](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)

原 glTF/许可已有本地已发布证据，本轮只取得原 buffer 声明的三个 `.bin`。官方 main 可变；本轮固定每件已收到的 body SHA 和 ETag，完整上游 commit 未知，不冒 immutable revision。原料和派生 NPZ 只在忽略的 local-only；Git 仅源码和小身份清单。

## 实际调用边界

`prepare.py` 对冻结框架归档逐 SHA 核验后，只提取并执行 `generation/src/gencuda/trellis_cpu_texture_export.py` 中 `require/corner_geometry` 的原函数 AST。未导入完整 exporter 或 Torch/trimesh，未加载 Shape Flow、稀疏解码器、o-voxel、原 A100 私有操作数或模型权重。原 source/数学函数身份见 SOURCE_PROOF.json。

实际网格按 glTF 原 POSITION/indices/静态 node TRS 读取，再变换到右手 Y-up 米制；所有索引、有限值和原 POSITION 声明边界实核。没有焊接、过滤或补面。边界边 WaterBottle582/Corset4522/BoomBox1092，非流形边和退化面均为0；保留原 index/seam 拓扑，不能宣称封闭/水密。

参考点使用固定每对象 seed、三角面积权重和 sqrt 重心采样。保存原参考世界包围盒中心/最长边，以 `(world-center)/extent` 得到固定参考坐标。预测必须已经按同一参考变换表达，不做每预测重心/缩放/ICP 拟合。模型输出到此坐标的真实桥接仍未资格。

## 可调用评分 ABI

`score_once.py --reference-root <原RUN> --split dev --predictions <新私有清单> --predictions-SHA256 <清单SHA> --output <不存在的私有结果>`。

清单为 `schema=TRELLIS_PUBLIC_REFERENCE_PREDICTIONS_V1`、`split=dev`、`objects.BoomBox={frame:reference-normalized-glTF-Yup-v1,predictor_receipt_SHA256:<64hex>,points:{path:<绝对预测NPZ>,bytes:<实长>,SHA256:<实SHA>}}`。预测 NPZ 只能有 `points`，必须 2048×3、有限数值。参考 NPZ 的真实 body pin 在 REFERENCE_INPUTS.json。缺失、无效、跨 split、帧不符或 hash 不符拒绝；不得补位。每个计划对象保留2048点分母，失败 F-score0、Chamfer null，不以成功样本平均隐藏失败。

Chamfer-L1 明确定义为两向最近邻**未平方欧氏距离**均值的一半和；[原作者实现定义参考](https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py)。F-score 为固定参考最长边1坐标下、距离含等号≤0.01 的精确率/召回率调和均值。这里2048点采样、坐标和阈值是独立新配方，未继承正式 A100/T10 评分器资格。

CLI 保留预测收据 SHA，但不会因此自动认证模型来源；实际预测/适配器/输出坐标与统一训练框架 checkpoint 的绑定须由后续真实 caller/独审证明。原生解码输出应复用同框架 export 的真实 vertices/faces，再按冻结参考变换采样；当前没有这种预测，不生成模型分数。CLI 内60秒是采样前的检查，不是操作系统硬超时，正式运行由根沿已有 bounded CPU wrapper 调用。

## 本轮真实验证与下一步

原 unique request `81416b29dcc443d8aa0dfaf0952d2bba`，三个新 GET/1019412字节全部完成，WindowsJob7.546s、2CPU、2GiB process+Job commit hard cap；峰 commit857018368字节、OS峰 working set64942080字节。CPU limiter 为真实 Job affinity，不宣称 cgroup。

`verify_inputs_once.py` 已实际只读三份真实 NPZ，核全 SHA/shape/有限值及 train2/dev1 隔离；新增缺失 dev 预测 CLI 测试实际退出1，保留1对象/2048点失败分母、0模型预测。原 prepare 的 self/平移/NaN 控制只作指标工程校验，不是模型效果。审计是本机输入检查，不是目标部署。

原唯一 once/verify 结果目录均已存在，**不重跑**。后续最短动作是根绑定一个真实 TRELLIS 解码预测和冻结坐标转换，由新 scope 提供上述 NPZ/原收据后调用评分；如先训练，仍缺目标 Shape Flow 生成缓存、rank2/alpha4 适配器与冻结边界、真实 optimizer 更新及恢复。原 T10 FAIL/UNKNOWN 和正式17验收结论不变。≥5真实可派任务目标本轮贡献 READY=0。
