# T04 render72 单 A100 原生训练数值/连续/恢复源码

源码独审后授权一次独立 A100 开发配方。执行入口现场检查 GPU5、完整模型文件及运行时。海光派发和正式验收仍未授权。实际先决native144已独立PASS/pub7eddf331，PLAN精确冻结其public result以及A100原staged binding/native-identities私有pins；不重做native，不读/传dev/blindgold，不读旧weights/initial/checkpoint body。

原8卡BF16/eager/GCfalse/LoRA16a32/drop.05/freeze边界作为正式参考保持不变。新配方显式world1/micro1/accum4/globalbatch4，原8卡globalbatch32；原4096、优化器、scheduler(max_steps25)、4个loader workers、train72原完整图文不变。新entry调用冻结404的hygon_vlm_recipe.build_launch，只用独立callback替换baseline；不改404、原domain witness、旧源码或旧失败结果。新scope自身连续开始的504 trainable初始值和完整RNG保存在私有initial.pt；replay只恢复该新自有初始，不冒旧初始继承。

一次普通source-only SSH安装后，单900s epoch依次运行三个真实新native进程：continuous25=100micro；replay前2step=8micro（max_steps仍25，nativecallback在2正常停止，避免warmup首步lr0掩盖问题）；本scope checkpoint20原生fullresume到25=5step/20micro。全部成本分母32optimizer调用/128micro，不把附加重放/恢复隐藏到25里。原Swift完成FWD/BWD、clip、AdamW和checkpoint/RNG恢复；观察器不实现训练loop或优化器。

首8实际完整forward logits/loss、前2 pre/postclip全部504 gradients、before/after trainables/optimizer/scheduler保存全typed私有参考；第二进程逐一16份比较，记录exact与CPU tensor逐块maxabs/relativeL2，无GPU F64。第二步必须非零参数变化。保存10/20/25全7件single-rank state（包括完整rng_state.pth）；恢复在native_load_rng_state实际返回点核全部Python/NumPy/CPU/CUDA状态，核全部20尾部输入/forward前RNG/loss以及最终adapter/optimizer/scheduler/savedRNG。72原行由原native sampler跨epoch，实际loader72/18update每epoch；真实entry返回、checkpoint epoch以及skip8都核定，不沿用旧800/8公式。

private capture<=32GiB、单文件<=3GiB、磁盘free>=40GiB。保守full4096 logits可能产生较大IO，实际吞吐/耗时/峰值UNKNOWN；900仅硬上限，不承诺完成。没有PC tensor落盘或任何HTTP/服务/隧道/新凭据。

固定A100GPU5/UUID GPU-19c8d012-5264-d041-a10c-48aa94938099/PCI00000000:A7:00.0，CPU6..9，避source10..15。root05:22:43UTC点时free40442/40960、0compute是线索；每阶段现场原qualified guard仍核free>=38000/no compute/UUID/PCI/合作锁。沿原qualified RSS解析器，另施本scope sampled64GiB软停止（不是hardlimit），host available准入32GiB/运行8GiB。最多>=5秒采样GPU5利用率和memory，生命周期窗口另列，不把分配/进程存活当busy，不声称640持续训练。

clock：admission100/phases790/super810/held825/tree835/hardtail845/post最迟855/producer860/reap865/独立reader<=35且<=890/reap895/whole900。没有新epoch或自动重试。原held+registry清理分别attempt，finally不可吞硬alarm，完整outcome后撤销。failed阶段保留实际progress与32/128全分母，任何失败不发后续phase。

submit已预接新process consume_once -> 一次普通SSH read_terminal，只读本次源/metadata/已存typed比较统计和context/owner/cleanup/post。不重算tensor math、不重新encode或重读assetbody，不把producer FAIL升级。parent原RSS解析FAIL request889e2c96...保持；原ownresume PASS只作source provenance，不重跑。Hygon候选1087656/a14r3n14仅资源映射；其model/404/native资格均false，必须另scope准备后才允许，当前Hygon_GPU_GO=false。

已实现接口待root/Astra独审，source_READY不等execution/ModelREADY；实际first8 capture大小、singleGPU峰值、真实跨epoch恢复数值仅由未来唯一actual判定。入口submit_once.py --publication <本包publication>；当前允许根协调者发布后唯一执行。verify_package.py --prepared只stdlib AST/MF及冻结metadata，不执行codec/fixture/Torch/Swift/model/native/GPU。所有rawIDs、输入、gold、tensor和weights留R私有。
