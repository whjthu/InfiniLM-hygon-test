# Kimi 同轨迹校验及真实动态到达实验

入口为 `launch_Kimi_matched_r8474.py`；源清单兼容名为 `Kimi-matched-manifest-r8472.json`。同目录保留已锁定的旧 loader、broker 和依赖，供校验复用，旧 launcher/协议不是本次入口。源文件中的编号和实际执行编号不完全相同，以 manifest SHA 和当前任务句柄为准。

BW1000 已唯一启动 910375.588。不要在原机器重跑入口、重复创建 watcher 或在观察超时后重新加载。另一集群须先迁移锁定依赖、核对模型/设备/网络、改资源和路径并生成新的执行身份；这些私有父对象和 SDK 尚未作为 BW1100 可独立构建产物验收。

完整 logits 的 CPU argmax 参考与 token-only 候选严格使用相同调用轨迹。单独保留 B16/M4 历史参考，不能把它当任意 batch 的参考。先保存实际输出及首差，再判断失败。回放只提供数值证据；真实 live epoch 才有请求时间线。动态窗口的 post-first 输出率与逐用户 decode、TTFT、ITL 分开，不能标成纯 decode 硬件吞吐。所有服务卡共 32 张。
