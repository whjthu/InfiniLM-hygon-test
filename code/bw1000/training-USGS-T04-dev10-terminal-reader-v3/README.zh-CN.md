# dev10 v3 原终态独立 reader（准备态）

仅消费 exact v3 namespace，复用已发布 v1 collector/离线核验的30槽输入与原生返回、raw EOS/cap、source/held-owner/退出门禁。原 v1/v2 原失败不覆写、不重读。

ROOT_GO/READ_GO/GPU_GO/Model_GO 默认 false。Root 在实际 once 后写新独占私有 context，填 request_id、publication、epoch、source_manifest_SHA256、claim_SHA256、namespace、original_client_returncode/outcome；当前 claim 中 null 不能执行。Root 发布/采纳 reader 后通过原 ordinary SSH 注入 CONTEXT、READER_CLAIM、READ_DEADLINE；不新增 launcher/服务。

单60秒含安装/传输，2CPU排除0/1，仅已知原小JSON与最多30原返回/3预测文件；原始读取总上限6MiB，私有bundle8MiB，默认每JSON256KiB、每return64KiB。0Torch/SDK/设备查询/状态张量/gold。只有同host/boot原supervisor/child/registry的PID/startticks已退出才读预测。未提交child但有control原锚的已知失败只消费终态小件，30槽保持UNOBSERVED，post UNKNOWN不猜clean。

新post独立核对保存的 transport 链metadata，而非信任单一passed标记。count=1必须是同一saved supervisor的directparent，唯一sameUID1001→两级root0链，前后PID/startticks/PPID/身份metadata一致，根有FD目录0500；exe始终UNKNOWN、传输FD未检查。其余同UID FD完整且无选中GPU/KFD持有者。count=0按实际完整可见路线核。不会声称全部FD clean或GPU独占，不以该准入作性能/正式17/Model资格。

离线命令：`python -B read_original.py --bundle <private-bundle> --bundle-sha256 <actual-SHA> --context <actual-private-context> --output <new-exclusive-private-result>`。该reader不评分；后续score_after需另行授权且完整30输出先封存。

当前只做源码/AST；没有远端读取或复跑旧fixtures。按v3实际transport字段封版；root负责独立审阅、注入真实context及实际调用。
