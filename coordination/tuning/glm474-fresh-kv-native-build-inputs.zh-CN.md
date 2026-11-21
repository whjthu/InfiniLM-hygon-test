# GLM fresh K/V 独立 Native 构建输入

`tools/prepare_glm474_fresh_kv_native_build.py prepare` 需要原 gfx936 编译的独立目标终态收据及其 SHA，核验证明、目标原始编译收据和 `fresh_kv_pack.o` 的身份。工具还按固定 SHA 读取 CPU308 实际构建收据与 Native 库，核对 30 个沿用对象，再按已发布调用侧源码写入新目录。写入失败时保留原目录作为对账证据。

按 CPU308 的实际编译命令在原镜像、原 include 搜索次序和原工具环境基础上执行两个 HOST fresh M。新 `src` 只放两个实现文件和三个新增 include。声明由 CPU308 原路径提供。`verify-dependencies --phase fresh --recipe-sha256 <原准备输出 SHA>` 要求两个 fresh M 均保留原依赖及字节，只增加冻结的候选文件和 CPU308 已登记的系统头。此后执行两次 HOST 编译，再以 `--phase actual` 核查实际 MD。两组均通过才用生成命令链接，保留 32 个原对象位置的相对顺序，在最后一个对象之后、provider 库之前加入实际 gfx936 对象。

`verify-link --recipe-sha256 <原准备输出 SHA>` 再次核对 MD 与 M、原 281 个强导出、两个新增候选导出、NEEDED 顺序和 71 个 provider 的路径及 SHA。保留原 Native。此处没有显式 Runtime 或 GPU 调用，`ldd` 的隐式初始化记为未知。正式 HOST 构建仍需同一次已发布 CPU 范围和实际目标环境。GPU 逐位、红区、真实 AR 三组配对和服务验收按各自门槛单独执行。
