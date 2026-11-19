# InternVL 三分片 zlib v2：已终态失败后的新范围，仅源码

全部 GO=false；作者完成源码与静态校验，不是独立审核或实际资格。原 v1 req328b0f23c38949f1a9149f0006e48b9b 的三 receiver Slurm 失败终态、三 source PID ABSENT、三 target-host PID ABSENT 已有冻结公开证据。PRIOR_SCOPE.json 绑定原 FAIL 和最终闭合；入口在任何 helper/SSH 前校验本次 publication 中两份精确 bytes/SHA。旧 v1 源、目录、partial、结果不改。新 v2 label/namespace 只用于这次新修复范围，不能重播 UNKNOWN。

## 唯一修复与证据

原03拒绝常量为 EXACT_ACTUAL_STEP_RECORD，旧代码以整个 scontrol stdout 行数必须1作为门禁。原 stdout 没保存，不能推断是0行可见性延迟还是多行。v2 将每次实际 step 查询原 stdout/stderr 有界私有落盘后再解析，只选择 exact StepId 的唯一记录；重复立即停止，0匹配可在同一实际句柄上最多查询3次（每次 subprocess timeout<=5s，总本机 monotonic<=15s，且仍在原 T0+180 admission 内）。每次原字节、SHA、返回码、行数、匹配数、实际耗时全部保存。3次仍0匹配停止，绝不以 requested 时限冒 actual TimeLimit，绝不重新提交 srun。

PC 与 target host 的控制字节在 JSON/stage/context 校验之前保存私有文件与SHA，包括失败终态、空/截断/非法消息；超过合法64KiB的首越界字节也留作有界证据。新 compute-ready-observed.json 在真实 compute READY 的 owner/context/numeric step 核验后、scontrol 可见性查询前保存。成功 receiver-ready 与实际 StartTime/finite TimeLimit/end 门禁保留。

失败独立 consumer 仍是新的 read_result.py 进程，每arm最多1条普通SSH与1次 exact sacct（总3）。同一SSH可读取本次该host固定 outcome.json/control-state.json/compute-ready-observed.json，每件64KiB、每host总192KiB；由完整 request/publication/epoch/MF/nonce/file_id 绑定的原控制或host receipt 解析原数字句柄，相互冲突拒绝，不再要求先有成功 READY。缺件/缺句柄仍 UNKNOWN，不猜PID/不扫目录/不查别的parent或label。新metadata不包含weights，不读取旧assets。producer失败仍失败，consumer不能升级；3files/11590blocks分母完整。实际query次数与原输出缺失单列，不把未执行query写成通过。

## 不变范围

三原shard完整SHA/大小共12,150,388,664B，11,590个1MiB独立zlib块，128B严格header，compressed cap、raw/comp/wire/whole SHA及EOF顺序均未改。common.py/endpoint.py/protocol.py/held_child.py/install_endpoint.py/node_entry.py 完整字节与原 v1 一致。原 rename后FD/path重新稳定取stamp、原epoch+85200覆盖终态close/save/control两修保留。

6个真实READY全部通过后才任何GO；源CPU10..15三组固定且无替代，receiver固定原3个parent/node，每件alloc4CPU/worker2/4096MiB/0GRES。source AS4GiB非RSS。PC只小buffer转发，不落权重，不新service/凭据/HTTP/tunnel。单86400秒及prepare60/admission180/payload84600/step85020/host85200/held85500/accounting85800/client85920/fresh86280/consumer86340/cleanup86370不改，不延旧lease。3次可见性查询只占原admission预算。

源一次wholehash与目标stream wholehash，不追加全body读。旧14、旧partial、initial362/364不访问，assembly=0。相同nonce marker只证明各observer见同内容与自身稳定FD，不冒跨HOST st_dev/inode相同或单observer见全3shard。旧assembly仍绑定原失败，需要未来显式revision；本包不修改它。资源/吞吐/Model/native/training全部未资格。

## 交付与执行边界

root独审后仅一致采用 PLAN/CLAIM root_GO/provision_GO/READ_GO并重封MF、普通发布；调用 `relay_once.py --publication <exact40hex> --source-manifest-sha256 <exact64hex>`。本交付不执行。verify_package只查MF、AST、原exact成员、预算/分母及发布先决pin；无fixture/codec/目标动作。private控制原文不公开，公开仅小字段和hash。
