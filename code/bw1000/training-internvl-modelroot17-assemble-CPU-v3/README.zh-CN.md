# InternVL 原 17 件模型只读组装

本包准备一个新的普通容器模型目录。原 .361 的 14 件与公开下载 shard02/03/04 留在各自的原始路径。HOST 在新目录创建 17 个零字节 regular 挂载点。Apptainer 把原文件逐件只读绑定到 /internvl-model 下的标准 HF 文件名。原始源文件的 inode、链接数、mtime 和 ctime 由核对程序保护。容器验证全部 17 个成员和只读挂载后，共同加载器才可使用 /internvl-model。

根协调者先接收并绑定三份独立的原下载终态和封存证据。authority 是待实际生成的小 JSON，包含以下字段。

- 顶层 schema=INTERNVL_PUBLIC_INDEPENDENT_THREE_SEALS_V1、本包 request 和 seals。
- seals 的键为字符串 2、3、4。各项包含 file_id、原 source 绝对路径、原 checkpoint 绝对路径、whole_SHA256、实际 receiver_rc=0、terminal_receipt_SHA256、独立实查的 independent_sealed_inode_verified=true、checkpoint_SHA256、checkpoint_bytes、final_stat。
- final_stat 的键为 st_dev、st_ino、st_size、st_mtime_ns、st_ctime_ns、st_uid、st_mode 和 st_nlink。根在精确原完成回执与最终 regular 文件核对后提交整个 authority 的 SHA。程序独立读取 3 个原 checkpoint.json 及当前文件 FD/path stamp，检查所有范围、确认字节数和封存身份。

用 `assemble.py inspect --authority <原件绑定> --authority-sha256 <原件SHA>` 做 HOST 只读预检。旧 14 件与实际 .361 原 outcome 中的完整 stamps 比较。如果一个或多个原件有无法比较的身份，打印需要另行授权定向流式 hash 的确切清单。预检的正文读取限于原 HF 索引。

待根采用 authority 后，设 CLAIM.CPU_GO=true 并填入 adopted_seals_SHA256 及采用的 adopted_public_shard02_checkpoint_SHA256，再封存新清单。只调用一次 `assemble.py prepare`，参数与 inspect 相同。返回原 MOUNT_PLAN.json 路径、SHA 和 18 个绑定项，每个绑定项作为单独的 --bind 参数。实际容器用 `assemble.py verify-mounted --plan <原件> --plan-sha256 <原件SHA>`，读取 /proc/self/mountinfo、准确 17 文件集合和 FD/path stamps。随后可在 HOST 用同一 plan 执行 verify-original 复核原文件及封存元数据。

本地测试只检查小型模拟回执与身份判定。当前采用 gate 关闭，目标安装、容器挂载及可用训练资产结论待实际观察。
