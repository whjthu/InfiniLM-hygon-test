# Treasury 三臂独立 GPU-v3 开发评测

新namespace training-qwen-treasury-explicit-dev3-gpu-v3；base=1087656/a01r4n14、old_domain=1087656/a01r4n16、domain=1103606/f13r1n10，各1DCU/8CPU/64GiB，三个fresh进程独立运行，不DDP/跨parent collective。base原底座、旧285checkpoint5、新328checkpoint5；同修订dev108、inventory07、greedy64、BF16/SDPA/batch1/4096和原14个模型文件（总17,545,907,231字节）。原native_generation和scorer等字节不变。只做开发对比，不宣称等预算因果、正式性能或盲测效果。

实际先决条件：330 nativeCPU、337/338/167三节点真实39文件/全部SHA门禁仍原样核；另新增63c2原IO gate341/342/170的actualsource11/claim/MF/intent/outcome/每arm stdoutstderr SHA+current actualaccounting3C0/CPU2mem4G0GRES。启动前真正执行scope.verify_prerequisites，不能用publicsummary或fixture代替；root独审authority97f0绑定全部实际原件。worker先bound_inputs核真实39文件，之后才HIP/provider/model。--contain和原嵌套RO挂载保持已通过三节点CPU的运行视图。

IO唯一修复使用已审53a2d2 controller和f659 helper：所有pre/GPU/post stdout/stderr独立保存为phase.stdout、phase.stderr。pre/post只读严格整个JSON对象11键，核node/actualstep/host_complete/fullFD/0holder/8PCI/boot，不筛行、不抽取任意JSON或忽略重复键。stderr完整留R并记SHA；未来root terminal reader须改用pre.stdout/post.stdout/gpu.stdout及各stderr，不能沿旧merged .log口径。pre/post workerPID/startticks/PCI/boot/KFD资格逻辑保持。

原v1三GPU输入门禁失败324missing；原v2三CPUpre C0后merged pre.log解析失败，0GPU但324missing；原v2 corrected reader的UNKNOWN保持不重播。两个旧namespace/source/原件均不变。本v3仅新source/claim，尚未执行。

单root原epoch1800；每臂校验/加载/生成最多900，GPU Slurm最多960整分钟floor，最多2880DCU秒=0.8卡时上限，不叫busy。GPU后留180秒accounting/post；late budget按同deadline缩水、记录actualworkintent，不重置epoch或增加预算；不等预算不称因果胜利。所有request进度及失败/截断/invalid/wrong/missing保全324、每topic每arm36。不得为填卡重复训练、扩大cap或prompt网格。

根普通发布后只安装MF精确源到R/results/training-qwen-treasury-explicit-dev3-gpu-v3/source，HOST绝对Python执行launch_once.py --publication COMMIT --claim-sha256 SHA --request-id NEW32HEX 唯一次。future freshparent/node/PCI/KFD/PID检查和原子namespace去重仍必需。UNKNOWN保原句柄不重提。执行后实际terminal+独立post及新文件名reader/scorer另验；signed_difference不改善或其他主题退化不晋级，不能因loss下降自动追加训练。
