# Kimi current public single-baseline few-shot quality preparation

固定 r444 的256题、gold及完整GSM8个train/MMLU5个同subject dev示例，不因Kimi答案、长度或历史重合换题。只换成Kimi-K3自己的explicit thinking=false普通请求，保持temperature0、4096输出预算、完整题干、无tools。单baseline256 HTTP，不制造GLM W2/W4质量配对。当前只授权一次4core8GiB300s CPU编码，无GPU/Native/权重/模型执行。

CPU源消费已真实通过572/576的r434 prepared与r428 runtime-v2，逐题真实protocol.prepare_agent_request(kimi_k3)、K3Codec、public_encode、history、validate_data、FIFO、反向重编码和32768容量验证。实际sampling dataclass为原公开源码；FinishReason只是CPU行政取消/retire叶子，没有模拟Native emit/reset或设备执行。原876/918/922旧套件不再运行。Kimi的tokenizer SHA、163840词表、163585/163586 EOS及所有prompt/body/ID SHA重新记录，绝不沿用GLM编码长度/160与96暴露分类。

三份已知历史题源仅以私有上传目录+SHA引用：Kimi GSM256、common fixedGSM32、shared historicalMMLU32。记录规范化题目重合ID及库存范围，不宣称超出该库存的全未见；原题不会被替换。全部题目/gold/prompts/token IDs/私有请求留R/L，Git只放源码、计划、pins与compact证据。

现source/CPU不运行公共HTTP，不发Model_READY。Future plan明确Native28800/service28200/GPU29400/keeper32400，256是全计划HTTP总数；原434六请求的4140/3540预算不继承。未来新的宽客户端须先一个32card普通公共434链实际READY、首HTTP和正常32owner/四node清理后，才由根确定是否条件并行分片，不能自动32×N。没有GPU资源认领/提交或质量通过。

CPU入口：`python invoke_CPU_once.py --manifest-sha ACTUAL --publication-commit ACTUAL`（只在R、source已普通发布后）；原成功容器wrapper将`CPU_inside.sh`作为唯一内层。结果为新 `/refactor/candidates/Kimi-wide-fewshot-quality-r448-CPU-v1/encoding.json`，无重试/覆盖。未知失败保原receipt先诊断。
