# I96 packed持久storage精确分配 r167

唯一改动：以实际CPU517 / Native `2e7c52d34c9ec6ccc6306517e77b76f56265e3309071979d43d425bc28254197` 的I96候选为父，仅将四个packed底层storage的`Tensor::empty`改成`Tensor::empty_parameter`。expert narrow/squeeze别名、shape、stride、Weight tp_dim、load、scratch、cache、算子和数学均保持原字节。bool参数仍未传入；它表示pinned host，不能当exact开关。

CPU包部署到 `/refactor/incoming/Kimi-TP32-I96-exact-storage-r167`。root审核发布后，在既有masked CPU keeper运行 `bash CPU_inside_r167.sh`。只重新编译一个HOST MoE TU，复用CPU517其余33个object（含原I96 HIP/provider/packed），严格链接新独立Native。420秒外层硬界，无下载、Runtime、dlopen、GPU、Model launcher或旧产物覆盖。新输出 `/refactor/candidates/Kimi-TP32-I96-exact-storage-r167` 必须不存在。

Binary closure证明当前Common Runtime中普通empty走pool.allocate，非CPU非零parameter走pool.allocate_exact。四个bank均256B整除，设计上每层消除135,725,056B class padding，92层合计12,486,705,152B（11.629GiB）；这是布局/分配策略账本差值，实测显存节省仍UNKNOWN，全Modelpeak/fit仍UNKNOWN_STOP。MLA cache rounding、I96逐层scratch、driver/CCL/算子高水位仍需独立闭合。

本包不构建adapter：旧r163 adapter绑定旧Native，不能直接拿它做新候选GPU证明。CPU成功只形成新Native receipt，不自动触发任何GPU或Model准入。后续如确需组件验证，须单独正确重绑adapter并保留4GiB floor；本包不提供提交器。
