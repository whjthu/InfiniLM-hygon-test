# Deep 融合的真实 Model 请求入口

此目录将 CPU902 的原 namespace 49-object Native 接到普通公开服务入口。父427的加载、实际 NativeContext/forward、生成、生命周期和 Common/SDK 映射保持；435 的 P1 流式文本路径固定使用。安装目录与已冻结的父目录均不修改。

实验模式仅由私有 worker 的固定六请求顺序 `0,1,0,1,1,0` 选择。外部请求不能指定融合模式。每次进入请求用原 typed NativeContext.reset 等待全部实际 owner/drain，随后才 clear/arm；所有原 forward 完成后读取真实8个 owner 计数。HostReturned 不是 GPU 完成标志。失败 poison 后不重试已失败的 reset；后续不处理新请求，交原 destroy 和外层清理。

范围为同一完整8489-token工具历史、普通chat、128输出预算、TP8/PP1/HC0/32K。预期一次8卡驻留，2warm+ABBA共6个请求。T1覆盖包括一token的prefill尾；严格命中期望从实际trace计算，不把大prefill shared/routed fallback调用数误当每层一次。真实 Model 构造、权重加载、候选命中、EOS/工具答案和性能仍未验证。

本次仅先做4核8GiB最多300秒的CPU胶水验证，无新HOST/GPU编译、不执行或dlopen Native。CPU包含实际修改后的serve、原generate及原NativeContext.reset/thread方法的8条recording-CABI场景，真实49object字节/依赖派生检查，以及同一CLI的非serve路径。recording场景不能冒充真实模型、GPU或HTTP通过。

设备配对还需固定客户端、控制器、CPU回执、准确资源认领并普通push，鲜活检查空闲后唯一启动。此目录当前GPU_GO与Model_GO均false，不因CPU成功自动视为已完成模型验收。
