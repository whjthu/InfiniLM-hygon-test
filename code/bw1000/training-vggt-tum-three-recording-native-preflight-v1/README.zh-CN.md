# T12 VGGT 三段录制九对输入的目标预检

本机现有私有阶段位于 local-only/training-vggt-three-recording-native-input-v1。只核对该阶段的 74 个已寻址文件（12,766,333 字节）和六个源码文件。原 bundle SHA 为 647b78e286687cc1d55670f64472fbe0f395c1b81cbeb928999fcf415f6a59f7，输入顺序 SHA 为 432012b6fce5fe874dd5883df184cd50f4f6d7dfb45f4b2699536839100c44d0。已有阶段可直接供根协调者准备已授权的传输，不需要再下载或重建媒体。生成方式仍是原 tools/stage_vggt_tum_three_recording_inputs.py 的三个原输入和唯一新目录。源材料保持私有。

目标 HOST 可运行 preflight.py install，将已比对的九对输入加入新的私有候选目录。--framework 指向原 .489 使用的 R/results/training-qwen-a68952c-swift-initial-CPU-v1/framework，--framework-manifest 指向同一路径 source/FRAMEWORK_FILES.json。--upstream 使用 R/results/training-vggt-native-provider-HIP-CPU-v1/upstream，--upstream-manifest 使用本仓库 code/bw1000/training-vggt-native-provider-HIP-CPU-v2/UPSTREAM_FILES.json 精确字节。--weights 使用 R/results/training-vggt-original-public-model-v1/model。阶段原件用 --stage，新的唯一私有目录用 --candidate。检查原 393 框架文件、69 个 upstream 文件和原 model.pt stat 后才创建候选目录。原模型按成功先例的身份检查，权重载荷未读取。

根协调者可在 1103606/f13r4n09 分配 0-GRES CPU 步，按 .489 的 DTK 容器和 /training-runtime/site 运行 preflight.py native --candidate <只读候选挂载> --install-sha256 <实际安装回执 SHA>。沿用显式空 ROCR_VISIBLE_DEVICES、HIP_VISIBLE_DEVICES、CUDA_VISIBLE_DEVICES，PYTHONNOUSERSITE=1，DTK 环境，设置 Python 包站点为已资格版本。原装 393 文件加六文件覆盖形成候选。容器内重新核对全部实际成员，然后通过真实 Torch CPU 调用冻结 VGGT 后端的输入转换与深度筛选。输出只含九对数量、格式、绑定哈希、录制来源计数和步骤句柄。Slurm 资源和原步骤仍需由根独立对账。

本机可执行 preflight.py assets --stage <已有私有阶段>，随后执行 python -B -X utf8 test_preflight_CPU.py -v。当前未实际执行目标步。目标 CPU 回执闭合后，需要把九对、三段录制和冻结输入游标接入新的单卡训练及检查点恢复包装，独立认领 25 个连续更新与恢复，并验证实际数值。旧 c556 和 bidirectional30 的原执行不在本范围。预检保留零次训练更新和 TRAINING_READY=false。
