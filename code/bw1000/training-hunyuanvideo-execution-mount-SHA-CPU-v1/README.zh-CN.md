# T09 执行挂载 wholeSHA CPU资格与实际 caller 接入

当前仅源码，root_GO/CPU_GO/GPU_GO=false，未读远端/权重、未提交Slurm、未访问A100/694原请求。不是GPU_READY。目标只是原执行节点f13r3n13的exact Transformer文件33,306,632,192B，完整SHA `71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86`。原v2 checker两段fingerprint/hash函数逐AST复制到original_hash.py；不复读旧hash终态或任何SDK。

根review/pub后安装小源码到 PLAN.root/source，在同一原epoch（包括安装）唯一运行：

```text
/usr/local/python3.12/bin/python3 -I -S -B <PLAN.root>/source/launch_once.py --publication <actualpub> --source-manifest-sha256 <actualMF> --request-id <unique32hex> --epoch <rootorigin>
```

CPU范围1103606/f13r3n13、2CPU、4096MiB、gresnone；900s全epoch，work最多epoch+780、Slurm最多840s且按剩余分钟向下取整，资源/节点/原label重名/intent去重先核。父作业UID数字为权威，RUNNING40/320和node归属真实核，不把预约叫busy。新namespace任何intent/outcome/workerintent已存在即拒，不因UNKNOWN改名重跑。stdout/stderr分离，heldPopen/PGID原字节helper锚先持久化，再等待及TERM/KILL/reap；原boundedterminal helper消费实际列头/accounting，不把非空RUNNING行当TERMINAL。CPU worker不创建子进程、只系统标准库，无Torch/SDK/native/model，Slurm4GiB内存资源并不冒RLIMIT_AS或测得硬峰。完整file FD在读前/读后/path全stamp一致，逻辑body只读一次、无payload写/下载/桥接PC。failed hash计数不能恢复时明确UNKNOWN/null，不能伪记0B。源stat/worker/CPUuser/system/wall/maxRSS都记录。物理I/O/cache/readahead/wire/GPUbusy未知。

这是新执行挂载文件身份资格，不能解除原before.pt/inputs.pt缺失或694transferUNKNOWN。CPU阶段禁止读取这些文件。保存成功workerreceipt不单独授权GPU：根后续单次运行audit_CPU_original.py（注入原CONTEXT及新≤60s READ_DEADLINE）仅小receipt/sourceclosure+原exactCPUstep的一次sacct，核COMPLETED0:0/2CPU4G0GRES/owned srun reaped/原fullSHA/原epoch。其输出成为未来root独立authority，精确bytes/SHA普通发布后方可用于新GPU caller。CPU退出证据明确是无子进程foreground worker与原SlurmC0，并非从其它login的/proc猜nodePID已退出。当前AUTHORITY.template全部null/false，不预填实际号或结果。

实际消费不是报告承诺：新caller-v3（旧v2不改）controller必须显式root/GPU采纳CLAIM_GPU.json并接 `--execution-authority-sha256 <published actualSHA>`，controller核新src/execution-authority.json SHA，然后

`slurm_command.command -> device_entry argparse -> run_replay.prepare -> inspect/final_suffix_identity -> execution_mount_receipt.consume`

传入同一authority SHA。新scope固定 `training-hunyuanvideo-sft-native-replay-v3`；源manifest14成员包含verifier/CLAIM，authority作为独立已pin小artifact。HOST原preflight仍需before/inputs原确切身份，不在HOST启动新wholehash；GPU节点阶段先实际消费新receipt，严格同f13r3n13/UID/boot/file(dev,inode,size,mtime,ctime)全相等，才零body重读复用完整SHA。容器只允许既有 `/refactor/results` 与HOST exact R/results 映射；无alias遍历/扫描。receipt改动/重启/挂载stamp不符直接拒绝，不fallback再hash或找替代路径。GPU原配方/数学/4096等原输入限制不变，缺before/inputs仍硬拒；此source不发新GPU。

未来authority先由根新CPU真实终态产生，随后才能单独审核/发布caller GPU claim和authority artifact；不是CPU成功后自动GPU。资源常规授权有效，无需新增服务器许可，但原被拒读取、未知请求均不绕过。check_receipt_CPU只测新receipt污损/挂载/原terminal与分钟边界，verify_package核AST/双manifest及actualcaller调用链，不执行模型或旧fixtures。

## 同包一次安装入口

根在同原epoch安装前生成唯一request/publication及已pub sourceMF，注入INSTALL_PAYLOAD_B64={epoch,request_id,publication,source_manifest_SHA256,package:{relative:base64bytes}}给install_and_run.py。包必须含MF及其全部精确成员（含caller纯源码，无权重）；整体原解码上限2MiB，单编码成员2MiB，外payload4MiB。真实root/CPU_GO须true、GPU/Model false、2CPU4GiB/1103606/f13r3n13/900s/33GB身份固定。新root/source排他创建，已有/alias/symlink一律拒，不给UNKNOWN换nonce重投。install-intent记录UID/PID/startticks/hostboot/原epoch/MF，每文件tick+exclusivewrite+fsync；仅安装MF成员+MF，再exec绝对Python-I-S-B launch_once.py，epoch不重置。失败仅在自己刚创建目录best-effort有界写install-failure；无SDK/GPU或子进程。

Installer工作alarm原epoch780、hard原900；exec前只移交原900硬截止，launcher接原850stop与cleanup，不遗留780默认信号误杀清理。Launcher持有srun阶段将TERM/HUP/INT/ALRM转可捕获中断，finally忽略第二次TERM/HUP/INT并保原900硬截止，按原helper回收自己的heldchild；后续accounting异常记录UNKNOWN/已见observations且不重试错误query，不变成无outcome。已读scontrol EndTime必须覆盖原epoch+900。check_install_CPU只测新安装metadata contract，不调用mkdir/exec/Slurm，不重旧fixtures。
