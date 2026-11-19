# InternVL 七 rank 原始 initial 的无损 byte-block delta（仅源码）

Root已独审采用单次无损差分范围，root/provision/READ GO=true；当前0次实际调用。GPU/Model/training GO保持false。没有读原始 weight/initial body、旧 receipt，没有 fixture、远端、Git 或新 service。原 1087656.362 及其源码保持封存。source complete 不等于 runtime/data/model READY。

## 唯一范围及真实分母

固定 rank00 为已有 362 新完整文件的只读 base；rank01..07 必须各自恢复原始整件 SHA，不能从 named parameter digest 推断整件相同，也不能继承 rank00 RNG。PLAN 的 8 条原件路径/大小/SHA 直接复制原 checkpoint-provision-v1 的冻结 PLAN，来源 pin 在 ORIGIN.json；不读其 private body 或旧 receipts。

| 成功时的独立分母 | 数量 / 字节 |
|---|---:|
| A100 原件完整只读及整件 hash | 8 件 / 1,398,575,192 B |
| Hygon 原 362 base 完整只读及整件 hash | 1 件 / 174,821,899 B |
| Hygon 新 namespace 输出整件 write/hash/fsync | 7 件 / 1,223,753,293 B |
| 不同块原始 bytes 的 wire frame | 固定 7 帧，每帧 <=4,194,304 B，总 <=29,360,128 B |
| 独立 fresh actual accounting | 新 exact step + label，1 次普通 SSH / 1 次 sacct，0 新 Slurm |

4 MiB 是整个 magic+length+JSON header+raw blocks 的上限，包含帧头。固定 65,536 B 块，尾块长度取原始余数；只发送不同的完整块，index 严格递增、不重复，块 SHA、帧 SHA 和每件原始 wholeSHA 都检查。0 个不同块仍有帧头。任何缺项、cap、hash、FD 稳定性或时限失败均 STOP，无 raw fallback、降阈值、重播。失败保留 attempted/verified/completed 实际分母，绝不将已验证部分报告为 7/7。

## 已实现数据链

relay_once.py 的第一次 main 语句建立唯一 900s epoch。只读本次普通 publication 的源码身份，两个 held 普通 SSH 运行短 -c bootstrap；8 B 长度+小源码 bundle 从 raw stdin 精确读取，不把权重放进命令行/JSON/base64。CreateProcess commandline <30,000 字符，bootstrap <=262,144 B，decoded source <=196,608 B。所有控制行使用 raw exact-length/单字节 line_fd，不混用 sys.stdin.buffer 和 os.read；GO 是 3 B。

target_host.py 先做 parent/label/namespace/fixed node 去重，持有真实 srun compute，node_entry/endpoint 完成 UID/node/CPU/目录/余量和 base metadata 检查后才 READY。login 仅小源码/metadata 和 RAM 转发，0 大 payload 文件。源只在两个真实 READY 后 GO 才读取 body，base 留只读 FD，8 件逐件一次完整读取/wholeSHA/stableFD，7 帧都在内存完成 cap 检查后才发布 PATCH_PLAN；此前 0 patch payload。

PC 校验 bounded PATCH_PLAN 后才 release target。PC 逐帧流读、独立帧 hash、转发，单块 <=1 MiB，0 大 payload 文件；计划和控制字节独立计数。目标 compute 只读已实传 base 一次并核 wholeSHA/stableFD，RAM bytecopy + exact block patch，每份只写到新 namespace 的独占 .incomplete，边写边 hash 验证自身原始 wholeSHA，file fsync/stableFD、新 inode、rename、directory fsync。base FD/路径从开始到全部输出保持稳定，不 chmod、不硬链、不改旧 14 或旧 partial，不读取输出 body 进行第二次 hash。每 rank 都是原 bytes 完整复原，包括自身 RNG，完全不解码 Torch/PT。

source/receiver/relay 正常终态后，relay 已实际接线以 sys.executable 新建 read_result.py 进程。consumer 验本次原 receipts SHA、同一 context、8 个 source 全件 pin、7 帧三端 SHA/字节、目标 base1 + 新 outputs7 的整件 SHA、file/directory fsync/stableFD、新 inode、held 正常回收；继而 fresh_accounting.py 实际一次只读普通 Hygon SSH 执行 sacct exact 新 step+label，核与 producer 终态一致。不是读 producer 自报代替独立 accounting，不是未来 TODO；本轮未调用任何 producer 或 consumer。

## 资源、依赖、单 epoch

A100 源：2 CPU affinity、排除 0/1、nice>=10、AS<=4 GiB（不是 RSS 声明）、0GPU。Hygon：parent1087656、a14r3n04、HOST Python3.12；alloc4CPU/worker2CPU 排 0/1、mem4GiB、0GRES。alloc4 不代表4 busy。目标新输出需 >1,223,753,293 +64MiB 可用空间；statvfs 不冒 quota 证明。PC 仅 RAM 桥/小私有 receipts。stdlib，无 Torch/SDK/模型加载/公网下载。

source RAM 约 base175MB+最多28MiB frame+暂存；target RAM 约 base175MB+单份copy175MB+<=4MiB frame，均受4GiB AS 限制；RSS 实际只报告 ru_maxrss，尚 UNKNOWN。CPU、I/O 和吞吐尚未测量。

预算沿已资格 362：准备<=25s、双端安装/准入<=75s；唯一 step 660..780s，取整分钟且 end<=epoch830；worker payload 到 min(epoch800,step_end-15)，host835，held cleanup855，producer accounting875，client885，independent fresh query<=min(begin+25,epoch892)，held SSH cleanup<=min(begin+29,epoch894)，consumer closure<=895，whole900。剩余不足就失败保留 producer 原件，不延 epoch、不新提交、不重播。source/target/relay 失联和未确认终态保持 UNKNOWN。producer 原有 exact-label accounting 的有界半秒等待仅到875，不新增 scheduler/poller。

进度每 >=5s 或 >=16MiB 才持久化，普通 wait 可以0.5s；进度 save 的 JSON/write/fsync/rename 单列成本。源 disk read/hash/compare/framebuild/pipe blocked、PC read/hash/write blocked、HOST read/srun write blocked、compute base read/hash/memorycopy/patch/write/hash/fsync 分开。并发进程成本不加总成网络独占；syscall 成功返回逻辑 bytes 与物理网络/disk bytes 分开，后者 UNKNOWN。capSTOP 时未完成 hash/compare 的输入计数保留部分；块 hash 的 verified counter 仅代表已通过项。

直接复用原 sealed rank00 scope 的 common.py、held_child.py、fresh_accounting.py、node_entry.py（精确SHA见ORIGIN）；新 endpoint/codec、relay/host/install/reader 的必要协议适配都在本目录。普通 Hygon helper 只冻结已资格路径/SHA，准备不读取密钥/连接配置。原362 public result 与现存 base 不重查；本次运行时读 base 是 delta 所需只读完整校验，不更改其原资格记录。

## 判定边界

仅当 8/8 source、7/7 frames、base1/1、新 outputs7/7、两 held SSH 正常终态及 fresh actual accounting 全部成立，closure 才 PASS。性能收益 UNKNOWN，可能 capSTOP；不能提前声称数 KB、加速或成功。只恢复七份原件，不证明剩余模型 shard、全部26件、native/model/training 或正式质量验收。

root 独审并普通 pub 后才可将 PLAN/CLAIM 的 root_GO/provision_GO/READ_GO 一致启用及重封 MF；GPU/Model/training 永远 false。入口 relay_once.py --publication <exact40hex> --source-manifest-sha256 <exact64hex>，只允许1次；不要现在运行。verify_package.py 仅 source/AST/MF/metadata invariants，绝不运行 codec fixture。
