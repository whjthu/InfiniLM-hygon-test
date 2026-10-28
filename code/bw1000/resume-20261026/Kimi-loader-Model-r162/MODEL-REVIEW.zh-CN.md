# r162：一次必要Kimi加载的具体待审批闭包

已冻结源码与本地CPU接口合同；**没有提交GPU/Slurm，没有运行launcher，没有SSH/Git，没有生成实际GO文件**。CPU511已过不等于Model合格；root审阅这一具体源码后决定发布、实际CPU import/preflight及GPU执行。

## 独立必要性与固定范围

Native加载历史1180–1408s尚缺registry/CPU/faultIO归因。r162使用已CPU511构建的Native `798c4bca…`固定binding-cache ON，测实际每owner registry构建次数/host时间与每flush进程CPU/缺页/IO；加载后马上运行C1四题和C8两组EOS小pilot，继续利用同一驻留，不为W13单开加载，也不承诺一次加载和旧冷暖不同的445有因果速度AB。

源基于445成功的136b完整launcher/runtime。目标TP16×PP2、cap8/pool17/pages68/context1024/micro1，32target+1draft；原节点/parent资源、CPU亲和、IBext_v8/shca、高速网络验证、capacity/head/key/显存floor、draft启动顺序、stop/退出/KFD审计保留。当前可用性由root原fresh-owner检查确认，源里没有冒充资源已空闲。未触碰保护的384/n17/n18。

## 明确源码变化

- `runtime/model-native-receipt-r9282.json`为CPU511数据副本，保留完整Native/34对象/依赖闭包；原远程receipt SHA `35453a2a74468d5bb28643d66263ab40467aa00a0aab8dbd767b819075751944`写在producer。数据副本本地重新序列化，其文件SHA由manifest另外锁定，不假称等于远程原始字节SHA。
- `parameter_loader_r9143.py`换为已CPU pointer/flush合同的r159观测版本，新增标准库observer；原shape mmap、A_log trim、2GiB flush、source生命周期和Native load/finish顺序不变。H2D API未装，明确UNKNOWN/null，Core不改。
- target_inside在Runtime建立前仅加固定cacheON、W13 OFF两个literal exports；manifest与目标启动前环境检查同样锁定。其它通信/LD/容器路径沿用成功136b。
- bind新增registry/cache/W13计数getter的精确Native dladdr校验；实际CPU preflight用nm验证这些strong T，不在CPU import里CDLL或创建Runtime。
- load后校验每local owner：state_dict first+finish=2，lookup/load调用数=实际flush数，finish=1。失配保留原完整failure报告并正常清理，不能把CPU cache构建直接称为Model成功。
- W13保持OFF，协议/math/TP/PP/原spec调度不变，加载前到protocol结束的branch计数要求candidate增量全0；没有新增W13 toggle接口或重编。
- audit保留原32owner、IBext_v8/无Socket、quality一致性、draft正常close、shutdown与post-KFD gate；替换成6case pilot完整性/分阶段加载归因，不再冒称12case GSM32或四个GPUprofile窗口资格。

## 固定真实请求

queue包含4个C1（各一题）和2个C8（各八题），W8/state、cap512、prompt不截、EOS不强制、W2pair沿用ON。20次请求执行、16个独立GSM题，C1四题是C8前16题的子集；不是新32题heldout、不是32并发、不是新集群吞吐纪录。

每题分别报告正确性、EOS、截断、prefill/TTFT、decode用户速度与总decode吞吐；不要求输出token逐位一致。stop紧随6case，无空跑等待。若deadline/显式stop导致未执行，审计保留缺失case和原因，不续播、不自动重新加载。只有完整20请求正确且EOS/未截断，才将`Model_pilot_qualified`置true；生命周期passed与pilot质量通过分开。

## 文件与待执行边界

本目录`model/`对应待部署`/refactor/incoming/Kimi-sparse-profile-Model-r162/`；producer与source pins锁定文件。`launcher/GPU-GO-schema.json`只是模板，实际`GPU-GO-after-source-publication.json`不存在。root可以沿原preflight_then_run入口审阅并最终授权；本agent不提供或执行临时绕过GO命令。

本地CPU接口合同覆盖cacheON的2-build/flush计数、坏计数拒绝、W13 OFF计数/误开拒绝、6case原题/答案和manifest/pins闭包，以及原Native batch/cohort/quality/resource逻辑未改。实际四node CPU import/强T/ldd-r、Model Runtime加载/全参数cache生命周期与EOS仍待root批准后的真实验证。
