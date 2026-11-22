# T12 VGGT 从原第 10 步恢复

原 `1103606.927` 的九对公开 RGB-D 相机头训练已完成 25 步，原第 20 步恢复也已独立核验。这次在 `1103606/f13r4n09` 上使用 **1 DCU、4 CPU、64 GiB**，启动一个新进程从原第 10 步恢复，只执行第 11 至 25 步。

请求 `adce8503c5a14bb5acfb4e851e6d337c`，目标目录为 `R/results/training-vggt-tum-three-recording-native-resume10-v1`。继续使用已验证的 398 个共同框架及覆盖文件、九对输入、原模型、DTK/Torch、相机头训练与冻结 aggregator。原 `.927` 真正的 `arms/continuous/receipt.json` 和原第 10 步检查点通过额外只读挂载使用。检查点为 2,594,191,507 B，SHA256 为 `8925acf4cbab54fe6023480b39f78482681946404fee49cae5a9b31a1f588fba`。检查点归档 SHA 以分块读取，不占用整份文件大小的 Python 字节缓冲。

工作负载保留全部 15 步的 loss、梯度范数、camera head 裁剪前梯度范数和数据游标。分别比较第 20、25 步的 camera head、优化器、放置、RNG 和完整边界摘要，并核对最终状态。原误差上限 1e-6。

唯一派发入口为 `launch.ship(publication, snapshot, transport)`。根协调者普通发布、核对新鲜设备准入及原保留后调用。总预算 1800 秒，GPU 步最多 1440 秒，容器最多 1300 秒，原前置、后置零卡步骤分别最多 45 秒。成功或失败都用原编号步骤及独立零卡清场对账。

`read_terminal.py <原请求目录> --actual-accounting <原三步七列 JSON> --independent-post <独立设备准入收据 JSON>` 使用真实终态重算同一恢复比较。新包当前只有本机验证，正式 CO3D、A100 性能和留出集仍需各自实测。
