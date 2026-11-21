# CMU 79/81 原生 ASF 正向运动学候选

实际完成940帧×30骨段，输出31节点（root+每骨段远端）的世界米坐标、30个骨段base、世界旋转和完整父子层级。私有结果在 `local-only/training-hymotion-cmu-asf-fk-candidate-v1-original/`；不提交坐标、原动作、官方归档、编译产物或官方源码正文。

`fk_matrix.py` 是独立NumPy矩阵实现；`oracle_harness.cpp` 实际调用官方代码的独立ASF/AMC解析器、`Skeleton::build_pose` 与 `get_world_bones`。官方reference只在本机私有目录中提取所需12文件，其中world-bones函数正文逐bytes保留，仅新增include/namespace外壳以避免构建viewer。没有运行OpenGL/SDK/模型。实际GNU++98命令与stdout/stderr SHA在私有receipt，未用Python实现冒充外部oracle。

CMU官方课程 [2013说明](https://www.cs.cmu.edu/~15464-s13/assignments/assignment1/assignment%201%20description.htm) 提供[参考归档](https://www.cs.cmu.edu/~15464-s13/assignments/assignment1/AMCViewer.zip)。该新代码来源HTTPS200完整6,527,132B；不是重下载旧录制。所有来源和12文件SHA见SOURCE_PROOF。官方源码只研究验证，不假设其具备可再发布的源码许可证；公开仓库仅本实现和身份清单。

约定：右手列向量，原CMU Y-up，不转到SMPL坐标。XYZ表示Rz·Ry·Rx；每骨段局部旋转C·M·Cᵀ，世界旋转由父旋转递推；base等于parent tip，tip=base+worldR·(原direction×length)。无DOF连接骨保持单位局部旋转。原ASF root position/orientation均0且order为TX TY TZ RX RY RZ，任何不匹配failclosed。官方info明确ASF/AMC为Euler；axis-angle段落是VSK/V，不能套用到此录制。

按[CMU官方FAQ](https://mocap.cs.cmu.edu/faqs.php)将原骨长和根平移同乘0.0254/0.45到米。参考viewer内部把两者乘0.45，它的内部显示单位再乘0.0254/0.45²作相同米制；没有按输出拟合缩放。原direction的有限小数并非完全单位向量，保留、不归一化。原限位是编辑辅助，不裁剪原动作。

实际C0、2CPU/2GiB Job、3.813s（含纯CPU编译和两个FK），全部28,200骨段观察。C++ float32 Pose端点最大差3.965e-7m；矩阵骨段跨帧漂移4.997e-16m，预先固定点/旋转阈值1e-5、名义骨长1e-6m均通过。存盘NPZ另以allow_pickle=False检查形状、有限值、连续帧和全部父节点base精确关系。它证明实现/格式自洽与参考代码一致，不证明实物测量误差或动作任务效果。

此31节点包含编辑辅助手指/拇指，不冒真实捕获的额外关节。原数据仍按CMU研究/产品使用及禁止直接转售条款，保致谢；原正式T08缓存和A100配方没有改变。Tencent SMPLH_JOINT2NUM/WoodenMesh配置、人体资产、关节映射和坐标匹配仍未建立；不输出伪135/201、不填充到目标维度、无训练/GPU/READY。下一步必须先核原135组成和对应body-model资产/坐标，再做独立重定向与数值资格。

两个once入口已实际执行，独占目录禁止重放。不得为再次生成证明而重新运行。GPU_GO/training_GO/formal17_GO全false。
