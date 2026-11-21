# T11公开atlas：真实target→input接线，原生特征尚缺

只读共同框架归档和两个已发布的公开atlas资产，不重GET、不读原T11权重/8RGBA/旧检查点、不运行模型或生成特征。

`target_join.targets_for_image(input_path, recipe=..., purpose=...)`是真实可调用的窄label hook：先按冻结binding精确匹配输入路径、RGBA文件SHA、object split，再读已hash目标NPZ；仅显式新recipe `sf3d-public-pbr-UV-atlas-v1` 接受，purpose必须train或eval-dev。返回原`foreground_material_targets`同结构的albedo float32[3]及roughness/metallic Python scalar，另返独立身份metadata，含每image+target+split+labelmode的SHA。

本机实际两个真实RGBA→target joins通过(.110s)，未重复旧512数学审核；实际将dev要求为train在读取其图/label body之前拒绝。并直接执行原缓存验证器的结构谓词，两个真实target-only dict都因缺`triplane_features`、`image_features`拒绝；仅这条原结构表达式，不冒Torch张量验证器已执行。没有Torch stub、零填充feature、手造feature width或encoder替代。

## 原生生产与消费的明确位置

共同`generation/src/gencuda/backends/sf3d_training.py` 58458B/dc57683c…：1050先require_cuda，1067加载SF3D预训练权重。缓存新建1117调用`_prepare_batch(model,image)`，1118 `get_scene_codes`，1121 `query_triplane`，1122 `_image_features`；1123–1131存两类实际feature及3类targets，415–428将真实tensor detach/CPU/clone存检查点，450–512验证形状、dtype、finite及模型实查width，515–524恢复。145–163由decoder与Beta heads消费这些feature和targets。

`sf3d_reconstruction.py` 34062B/aa0d1992…的 `_prepare_batch` 67–90不只是PIL：调用model.prepare_image、sf3d camera utils、硬`.to("cuda")`、model.image_processor。`_image_features` 167–189使用Torch bilinear224/CLIP常量归一化及model.image_estimator.model.encode_image。当前本机fresh find_spec(torch/open_clip/sf3d)全false；在这个已冻结入口及本机环境中，没有已部署的无权重原生batch/encoder链可执行。本轮没有尝试加载权重，也不为绕开依赖改成Pillow假native。

最小未来插入点：在CUDA/模型加载前验证新recipe、训练split及image/label身份；原1110标签选择仅在新显式recipe调用上述hook，原1117–1129 feature生产及原loss数学保留。现原config hash 319–331仅8个优化参数、没有label manifest，必须把train标签/输入绑定身份纳入新恢复合同，否则仅image相同不足以区分旧RGB启发式标签。不得把development标签身份并进training batch。此处只有可读的私有descriptor，没有改共同backend或假称hook已经部署。

## 两个实际阻塞

1. 原训练入口1063要求至少2个现有input_images；目前train只有WaterBottle一张，开发BoomBox单独保留。实际执行这条原条件得到拒绝；没有偷用dev或自动复制一张凑独立样本。
2. native model-bound features、宽度、CUDA/Torch运行资格均未取得。公开target NPZ只能满足3label字段，不能假装5字段cached batch。原2048随机3D query没有UV对应，per-UV512标签不能直接当其逐点标签；现接线只对应明确的新global uniform-UV atlas mean recipe，不能冒实景表面积或相机材质监督。

SOURCE_PROOF和私有source-call-trace逐函数SHA/行号绑定实际源。目标数据出处/CC0素材、CC-BY元文档和作者/修改说明沿已发布0e3c454资产，不改变原T11参考/审批。真实进展是source→input/target/split闭合及具体native缺口，native/features/model/training/READY均0。原8例数值/恢复未被继承或覆盖。
