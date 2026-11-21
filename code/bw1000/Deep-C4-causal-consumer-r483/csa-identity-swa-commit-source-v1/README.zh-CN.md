# Deep CSA 提交阶段跳过身份行复制

真实 r467 两个 AR 窗的 `commit_cache` 流区间分别为 14.559109 和 26.137104 ms。该区间包含 SWA store、compressor/indexer 的事务提交和同步。这些数值用来定位需要检查的调用方，尚无独立复制耗时。

在 CPU1113 实际构建采用的 CSA HOST TU07 中，提交行完整覆盖原 BF16 `[T,512]` 时，现有 `IndexSelect` 仍复制全部行。候选在单独进程显式设置 `INFINILM_DEEP_CSA_IDENTITY_SWA_COMMIT=1`，且索引逐项等于 `0..T-1`、槽数相同、准备张量为同设备 contiguous BF16 `[T,512]` 时，将原准备张量交给同一个 `PagedLatentCache::store`。原事务持有该张量直至同步。若长 prefill 裁掉窗口之外的行，或任何校验失败，沿用原 gather 路径。

单提交 token 的源码开销变化是省去一次 `IndexSelect` 提交、1024 字节 BF16 逻辑临时张量、8 字节 I64 行号上传，以及算子各 1024 字节的逻辑读和写。原槽上传、量化缓存写、验证读回、压缩器和索引器提交、最后的同步及失败 drain 保持相同。这些数值不代表真实 HBM 访问或已测加速。

```text
python -I -S -B check_local.py
python -I -S -B render.py <new-source-directory>
```

本机用冻结生成器提取实际修改片段并编译 C++ 记录器。覆盖未设置、启用和无效开关，检查 T1、环形窗口、混合请求、超过窗口、非连续张量和乱序/重复行。记录器对比完整 BF16 位、直通身份、无输入写入、临时张量与索引上传计数。目标 TU07 仍需新鲜 M/MD、实际对象与链接验证，再在同一原工作负载上核量化缓存、事务状态和配对延迟。
