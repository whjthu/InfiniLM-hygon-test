# 原BoomBox ORM的一文件补齐：真实PASS，原10GET失败保留

原scope终态已发布7e0bb12：开发ORM HTTP200，但Content-Length4,471,451超过当时技术cap4MiB，read/saved0。此次明确新请求e169e9032f7b4f03843117183351bc5b，独立目录/90秒epoch；只同一冻结官方glTF所指的同URL一次GET，严格核原唯一ETag、精确声明长度、image/png、无编码、自然EOF及fsync。If-Match为原强ETag；缺、错、重复ETag或size不符均在body前拒绝。不重跑原请求/其它文件、不换对象/代理/路由、不读原T11检查点或8RGBA。

实际新GET1/4,471,451B/Job C0/10.75秒，实际2CPU、进程及Job 2GiBcommit硬限；OS观测峰工作集271,839,232B，不说RSS硬限。原10GET FAIL及全部失败分母完整保留；累计11GET、13,509,250B，故不能说原scope后来PASS。

开发RGBA2048²及4,194,304均匀UV texel真实PBR参数解析完成。保原WaterBottle train输入/目标逐SHA不变，仅新增BoomBox dev；全4张原PNG SHA均不同，解码为canonical RGBA后4个像素SHA也不同。它证明exact-content去重，不证明语义或近重复/OOD独立，同一作者Microsoft制作风格仍共享。所有纹理、UV、派生patch/视图按完整对象共组：train1对象、dev1对象、blind0。

`complete.py`复用已发布PBR math及target-only reader，开发原native foreground_material_targets纯NumPy函数实际调用1次；训练侧不重算旧启发式。两对象目标albedo(1,3)/roughness(1)/metallic(1) float32有限且[0,1]，固定UV各256共512。开发启发式与glTF参数UV均值差为albedo .134398、roughness .733374、metallic .008772；仅表示标签语义不同，绝不是模型效果MAE或原参考已改善。

`audit_labels_once.py`不导入generator或PBR helper，独立用Pillow.getpixel+Python标量sRGB函数以及原G/B通道公式，核全部512个落盘UV标签、原UV次序和4个像素身份。真实512/512通过，最大float32差2.8752e-8<冻结1e-6。只读审计不生成新标签、不重做模型/编码。

许可沿原官方Khronos [WaterBottle](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/WaterBottle) / [BoomBox](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoomBox)：Microsoft2017，素材CC0，LICENSE/metadata文档CC-BY4、logo/trademark除外；保原文、来源及修改说明。glTF [规范](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials)规定baseColor解sRGB、G=linear roughness/B=linear metallic。这里是作者定义shader参数和uniform UV atlas统计，不冒实验室材质测量或相机可见表面积标注。实际原main各body固化，并严格匹配旧ETag；完整上游commit仍UNKNOWN。

目标文件只提供原SF3D缓存所需3项label字段，未产生triplane_features/image_features。原生模型预处理/特征、真实渲染视角与UV对应、完整盲组及新A100数值/真实更新恢复仍欠缺。原T11参考/传输审批不变，原T07 TorchVision依赖也未解决。资产与512标签已闭合，native/feature/model/training/formal17和READY仍false。

私有结果、4图身份及完整train/dev路径在 `local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original/complete-data-binding-private.json`；所有原图和label留local-only，不进Git。两个once已实际完成，不得重放。
