# GLM fresh K/V 显式 decode 的 Native 构建

显式 decode 路径重编 CPU308 的桥接单元 u00、attention 单元 u15 和 prepare 单元 u17。一次构建保留原链接顺序，复用 29 个 CPU308 对象。3 个 HOST 对象和已独立核验的 gfx936 pack 对象进入同一 33 对象链接。

正式目标构建使用 `tools/run_glm474_fresh_kv_explicit_ar_once.py` 和该次发布的认领。运行前须取得原 pack 请求的实际终态和已核对象 SHA，确认 O01 节点归属及原步骤已清场，再为此独立 CPU 请求保留唯一启动入口。worker 检查 4 CPU、16 GiB 认领、零卡作用域和同一 1800 秒绝对预算。

3 份 fresh M 都通过实际依赖校验后才编译。每个 HOST 编译生成的 MD 必须与其 fresh M 一致。链接验收要求原 281 个强导出、两项 pack 导出和新的 `modern_glm_global_fresh_kv_decode_r474`，以及 CPU308 原 NEEDED 与 71 项库 provider 绑定。目标库 SHA 和实际运行时符号归属用于后续设备数值验证。

本机回归使用冻结编译计划、源 pin 和模拟收据核对阶段顺序、依赖拒绝及新增导出。gfx936 pack 的原终态、此候选的实际 HOST 编译和 GPU 验证仍待取得。
