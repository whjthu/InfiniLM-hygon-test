# r159a：CPU503 parent专用registry构建入口

就绪范围：已在本地用实际503 receipt做14项纯CPU fixture，未远程编译/链接/加载。root可在已有407/a14r2n20的**c4/8GiB CPU步骤**单次运行，和W13directdecode161 CPU步骤分别使用独立目录。外限175s/内部总预算170s；入口不自行Slurm/SSH/Git，不GPU编译、无dlopen。

```text
bash /refactor/incoming/Kimi-loader-registry-build503-r159a/CPU_inside_r159a.sh /refactor/candidates/Kimi-loader-registry-build503-r159a <root审定source-manifest-r159a.json的SHA256>
```

out目录必须全新。若root将incoming放到另一位置，脚本通过自身路径发现source，无须改文件。

parent固定：`/refactor/candidates/Kimi-MoE-phase-component-r155/CPU-build.json`，SHA `c2174202d50d2c7939bc169f318d71c53ea630c25684e75d8880a154d94374dd`；Native `d559f6a60081b50aa43dcb5a442119c38d62cc757a5c3d35619c599dcef2dc2b`。

## 已消除的schema/链接误用

- 503有31个`unchanged_object_SHA256`和2个`new_object_SHA256`，没有旧式all_linked字段。严格合并恰好覆盖实际Model link的33个.o。
- 503最后一个shared命令是component adapter；现在按`-o == parent.native`精确选Model link，拒绝只有adapter/重复Model link。
- 503没有worker编译命令。`compile-templates-r159a.json`来自qualified r9212d真实`CPU-observed-a.json/build.commands`，原receipt SHA `dbaedb154c3ea77e9f787b3d31bcac4fc9abb14306971315d56cbb2b24e58527`、qualified Native `2e5c21ba5738ce3711e1dfd2c033ebb641f99fa934268af3757b28010127ec75`。不是由Model对象名猜flags。只替-I/source/-o/-MF，其他实际host flags保留。
- 重新校验503实际new_MD（provider/adapter/moe），所有原始math对象SHA完全保留；W13/W2 provider `7f6ccd4b…`与moe `79185c1d…`均锁定，只重编worker/weights两个host TU。
- 不再要求full olddefs⊆newdefs。强plain public C API和实际required symbols须保全；弱inline/template移除（包括nlohmann vector<int>C1/C2）只分类记录，不假装公共ABI损坏。另验证DT_NEEDED保持、readelf版本信息、实际`ldd -r -v`引用/版本完整闭包。没有临时GPU后放过门槛的流程。

## 候选实际改变与输出

缓存/owner-generation/finalize验证复用r9212d；新增每owner registry host时钟。构建后CPU-build.json保留原parent33对象、实际编译/链接命令、新MD、导出差异及math SHA。private Weight symbols局部化，不覆盖SDK public Weight实现。原W13/W2数学对象不变，不更改W13默认OFF/ON开关。

没有集成H2D可选Core补丁，Core不改。未来Model加载若带`parameter_loader_r159.py`和`loader_work_observer_r158.py`，可记录每flush process CPU/fault/IO/registry；未安装H2D API输出UNKNOWN/null。不要为了它延迟本CPU闭包。CPU构建通过仍不是Model/GPU GO，后续只在有独立必要性的Kimi加载中piggyback原计划的W13 OFF/ON对照，W13专用Model继续PARK。

本地fixture验证实际503、33对象、adapter误选拒绝、math SHA变化拒绝、缺MD拒绝、CPU flags来源、弱C1/C2不误阻断和required C API缺失拒绝，共14项通过。真实Tensor host编译和依赖闭包由这一CPU步骤完成；不把fixture叫实际Native构建通过。
