# Production collector variable repair v3

仅修已完成 hc-pre-deployment-v2/collect.py 的局部路径变量：entry_report_path 始终指向 entry-report；hc_packet_path 只指向当前 HC packet。原 D2 源/MF、原 FAIL、所有运行原件均未修改。该 collect.py 继续依赖原同部署 support.py 及原 hc-pre-entry-v1/hc_consume.py，供后续新 immutable deployment 显式装配并重新绑定清单，当前不构成 GPU/Model 认领。

真实原件回归：

```text
python -B check_original_regression.py /absolute/Deep467-HCpre-1476-sealed-v1.json.gz --root-audit /absolute/Deep467-HCpre1476-root-independent-original-audit.json --output /absolute/new-regression.json
```

先验证69文件字节与根6fb1cb独立结果，再提取原 collector/support/hc_consume 及真实 entry/packets 到私有临时目录，执行原 collect 与新 collect。只把原 support.R 文件根重定位到临时解包目录；collect/hc_consume/JSON 值均非 mock。旧返回2d73637f末 HC packet摘要，新返回458748fc entry摘要；除 entry_report_SHA256 外完整输出逐data相同。临时原始JSON运行后回收，不入Git。REGRESSION.json保完整无token返回值，路径只规范回原/refactor。

0远端操作、0新CPUjob、0Native/dlopen/Runtime/权重/Model/GPU/HTTP；GPU1476正常、keeper1035失败、forced_cleanup/原TERM/post等历史不追认或回写。精确差异见 exact.patch。
