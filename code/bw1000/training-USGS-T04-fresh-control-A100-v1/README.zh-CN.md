# T04 USGS generic-control fresh25 独立新臂

仅源码准备，root_GO/GPU_GO=false；新namespace training-USGS-T04-fresh-control-A100-v1，原domain25已真实训练且其监督器FAIL保持，不重跑domain。原8A100 fresh admission在未来根审核发布后由原common入口实际执行，不能复用旧设备快照。监督器/Native/Git安装共用根提供唯一epoch900，Native820/stop850/cleanup880/alarm890/client895；held Popen身份、PGID/startticks/boot、registry/post沿用RSS-v2合格实现，不建service/poller。

同一396 base加5份既有overlay形成400文件。5overlay与domain原包逐字节相同，不改训练loop、模型数学、入口、processor或目标层。installer核每一原396文件和所有overlay/merged400，再真正exec本包supervisor/bounded_GPU_supervisor.py，强制其SHA2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f，传nested MF/CLAIM精确pins，监督器自身再作source_gate；不是只把RSS修复写在报告里。root采纳时须同时翻外CLAIM/CONFIG和nested supervisor CLAIM的GO，并刷新nestedMF、INSTALL_PLAN里nested pins/configpin、总MF。

control800原314277B/SHA564e7439...，45unique重复17/18次；实际8rank每rank100micro/25update，全局800分母。native45参考78022B/12e9f892...、control occurrence2356B/c9c69a23...独立绑定，仍每样本实际原生hash/labels/EOS/有效监督tokens记录。domain/control声明索引相同而输入/监督token不同，不能称同负载正式性能baseline。

真正加载原domain /run/witness/initial.pt 174808783B/SHA b574c04a3bdaaff7f061d7285dc0fc73f2cbadce4801fca97dbd00ad8772fa0d，经pair_initial_reference的551B/c5967f3e...核pairID/domain/rank0/global0/504/43646976/initial全部字段，再由原restore_initial_trainables在empty optimizer/global0下核全SHA/manifest/finite/name并复制504参数。原8rank真实named hash一致门禁保留。只设同seed不能替代这个加载。原语义核加载前后本rankRNG不变，不设置domain RNG；两臂初始RNG若缺原逐rank对照则保持可比边界UNKNOWN。CPU本机不读取initial.pt，不Torchimport。

配方保原BF16/b1/accum4/LoRA16/32/.05/eager/4096/GC OFF/AdamW与cosine/25updates。学习率仍原warmup_ratio0.03（已发生第一步lr0、第二步.0001），不误改成5步LR warmup。5warm/20measure是正式性能计时口径，本工程observer含额外hash/CPU状态采样，不作正式性能。checkpoint10/20/25保持，所有wrong/missing/失败/无rank0原历史失败不洗白。

未来唯一安装入口与原包相同：根将精确SOURCE_FILES及package小源码编码INSTALL_PAYLOAD_B64、绑定实际request/publication/MF/claim/epoch，执行install_and_run.py。该body只安装源码到新namespace并exec监督器，无复制模型/数据大体到本机。初始state由每rank原safe loader全文hash+CPUdecode，逻辑读取约2×174808783×8 B（另模型原生加载/状态checkpoint I/O记录实际成本，缓存/物理IO未知）。继承32..64CPU exclude0/1，8rank各OMP8/workers4；owned groupRSS256GiB是采样soft stop，非硬峰/硬RSS。私有输出预算2GiB，全部原证据留A100，公开摘要只小计数/hash。

本包check_control_contract_CPU只测新control/初始绑定分支，verify_package只AST/MF/实际driver及400字节闭包；不重旧RSS或domain fixtures。没有实际GPU调用、权重读、远端probe或Git动作。正式17项/Hygon/效果/恢复/performance仍false。
