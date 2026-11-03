# r444 原件离线最终消费者

仅标准库，读取已封存的真实字节，不收集远端、不轮询、不开 HTTP、Native、tokenizer 或设备。当前仅源检查；尚未消费在途 r444 的真实结果，不存在本 reader 的设备或质量 PASS。

此前没有 r444 的最终512请求 reader。复用已通过 CPU918 的 `scoring.py`（完整 SHA 固定）及 client-v1 的纯 `verify_trace/verify_native_calls`，沿418/425的原 body/wire→实际owner→token/EOS/reset→正常销毁/post原件审计口径，不执行 controller/collector。新增差集只负责8个互斥slot组合、原wire独立评分和全局覆盖率。

私有 index JSON 示意（路径均为 seal 内的原绝对路径，`/refactor/`和真实 R 前缀会统一；不得填根据信息推测的文件）：

```json
{
  "CPU918_archive": "L/GLM444-CPU918-original-inputs-and-encoding-sealed-v1.json.gz",
  "execution_archives": [
    {"path": "L/actual-slot0-seal.json.gz", "SHA256": "ACTUAL_ARCHIVE_SHA256"}
  ],
  "slots": [
    {"slot": 0,
     "watch_path": "/refactor/results/GLM-r444-fewshot-fewshot-20261104-a-s0-watch.json",
     "client_path": "ACTUAL_WATCH_CLIENT_RESULT_PATH",
     "run_directory": "/refactor/results/GLM-W4-fewshot-quality-r444/fewshot-20261104-a-s0",
     "accounting_path": "ACTUAL_SEALED_SACCT_TEXT_KEY"}
  ]
}
```

未来有真实结果时：`python -B analyze_original.py PRIVATE_INDEX.json --output PRIVATE_AUDIT.json`。当前任务未运行此命令，无 result 文件。源码检查只用 `compile(read_bytes(), filename, 'exec')`，不导入生产模块。

archive可为普通/gzip JSON；`files`沿现有 seal 的字典或列表，逐项有 `path`（列表才需要）、`bytes`、`SHA256`和 `base64` 或 `gzip_base64`。所有文件逐字节校验，冲突原件拒绝。accounting为原 `sacct -n -P -o JobIDRaw,JobName,State,ExitCode,...` 文本，前4列不可变。索引不得把不明终态写成正常；缺accounting/owner/post列为证据失败。多个归档可复用同一原字节，不能用不同版本覆盖同路径。

实际留存结构已足够，不要求热改在途源。每个完整slot封：原watch、client `result.json`，全部64个request目录的 `request.body.json/headers.json/response.wire`，run下16个 `owner-stage-rank.json`、16个 `lifecycle-stage-rank.json`，以及原GPU/client/keeper accounting。watch中的独立双node post包含KFD、已知PID、物理PCI及43843可绑定结果；保留失败watch/强制清理，不追认正常。初期/异常seal可不完整，但相应slot不会通过全终态验证。

冻结256cases/gold/body、实际prompt IDs/编码SHA直接从CPU918原283文件seal读取；不在Git复制完整输入或token。全部输出按原wire content独立重评，不信client summary。8slot分配严格 `(ordinal//2)%8`，W2/W4同题同slot，原512 schedule顺序/hash一一验证。固定256题/512请求分母，缺失/未验证请求不隐去；报告各套件准确率、Wilson区间、原预注册paired Hoeffding区间/McNemar、缺失结果最坏/最好边界。160个实际编码控制暴露题与96个控制等价题分别报告；两者仍保留在完整分母。不把partial的条件区间当全量noninferiority。

16owner全部检查实际profile/windowtrace/counters/finite、输出hash/EOS/reset签名、Native CAPI完整调用数量、原8GiB/4GiB门槛、无V census、4CPU互斥/不迁移、销毁和shutdown各一次。质量错误/parse/cap不触发生命周期失败。HTTP/token原件关联并不独立重跑tokenizer decode；此限制保留。custom chat GSM128+MMLU128不是全套benchmark，无预设非劣margin，无automatic-all-correct门槛，无性能/SLO外推。
