# CPU351 入口失败：仅本地诊断源码片段

当前351 Torch rawversion、实际 __file__ 和源SHA仍UNKNOWN；此前rc255/47B只读scope不重播、不换入口。历史141的2.10.0不能替代当前观测；330/328小终态亦无此身份。因此没有足够证据修正接受版本，本草稿保留原严格literal，不增加允许版本。

唯一改动在原HOST optin分支：在版本断言之前，把已导入Torch的raw/metadata版本、HIP/CUDA/git版本、__init__与version模块真实路径和有界源SHA写入独占torch-identity-before-guard.json，随后执行原guard。失败阶段记录绝不是host-main-receipt或108/324资格。只观察已有模块，不调用device/API、扫描sys.path、环境或shadowpath。诊断读/写失败仍fail closed。

本目录只有替换helper、证明和manifest；无新main、launcher、container入口、部署路径或CPU claim，不能直接调度，也不授权再次读取UNKNOWN scope。所有GO=false。父v1与native_generation.py全bytes不改，默认GPU guard/math/cap/input/provider边界保持原未合格状态。40KiB可写不证明任意JIT容量，不晋级GPUv4。

后续最多两步：根先审身份缺口及此片段；仅在另有明确许可且取得当前精确软件身份后，才冻结guard修正与HOST集成。当前不执行。
