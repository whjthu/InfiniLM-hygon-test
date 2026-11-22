# GLM fresh K/V 首段 T512 caller 试验

组件步 1103606.1026 的 T512 t128 吞吐为 t64 的约 1.752 倍，T1 相差约 0.22%。这份源码将 T512 路径限制在显式试验调用的首个完整 prefill 块。请求必须是单行 row0、past0、512 token，且模型已加载、持有有效流、只取最后位置的 logits。普通 prefill 和原 T1 AR 调用仍走原入口。

试验只包住 Native attention 首次到达 K/V 的 `expand_cached`。原 FP8 projection、位置和请求预检、后置 causal mask、历史缓存写入和状态提交沿用原源码。t128 前逐件核验 Runtime 所有者、设备、BF16、完整 shape、每维 stride、逻辑字节数和 2 字节元素宽度。四个原读写区间必须对齐、互斥、长度可表示。布局或区间拒绝时执行原 K192 拷贝、逐 head rotary 广播和 V256 contiguous 路径。实际内核 enqueue 失败沿用原错误返回。

新增 typed opt-in `prefill512=True` 返回 attempted、packed、rotary_layout、tensor_layout 和 address_span。原件保持默认调用语义。试验读取实际每 rank 的命中与回退。出现布局回退时仍保全普通模型结果及回退证据，在全部预期层真正命中之前不采用候选结论。当前本机无原 rotary 实现正文，已绑定其原 SHA。已知 k_rot 的逻辑形状来自 u17 原始 producer。真实 T512 行步长由实际 Tensor 实测，判定由新守卫执行。

运行 `python -I -S -B check_local.py --cxx g++` 验证实际渲染源码、typed 绑定、错布局和旧拷贝回退。渲染命令 `python -I -S -B render.py <new-output-directory>` 输出精确源文件和各 SHA。CPU 目标资格预计从 .918 重新闭合 u00、u15、u17，复用 .855 的真实 gfx936 对象和其他合格对象。仍需新 claim、目标 fresh M/MD、实际导出和 provider 验证。当前仅本机 C++17 与 Python 验证，零目标构建及 GPU 结果。
