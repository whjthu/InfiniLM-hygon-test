# OneVision EOS and sequence cap diagnostic

The sealed .5232 Val16 run reached 128 tokens on all 32 slots and recorded no EOS exits. Its native token ID sequence was not retained. This diagnostic does not ingest Val16, CP25, predictions, scores or answers, and never re-tokenizes prior output. It uses the same previously exposed CLEVR training PNG as the zero-GRES .5225/.5226 work.

On the qualified CPU model and v4 cache-position repair, it observes the actual Swift template generate invocation and the HF stopping list returned by the installed GenerationMixin. The first model forward is replaced by a sentinel preserving its original signature. The original forward is never called. There are no generated tokens or predictions. Actual effective EOS policy is split into model file declarations, live model and generation settings, tokenizer EOS, Swift effective generation config, HF special EOS tensor and native EosTokenCriteria IDs. Template suffix and stop words, EOS ID sets and native provider files are summarized by counts and SHA-256. MaxLength, minimum constraints and actual list class identities are separately reported.

The three controlled cases run the actual native criterion classes on CPU input IDs with an EOS suffix, an ordinary non-special suffix, and an ordinary suffix at the native length cap. If the original exposed input IDs are available at the correct width, they are used in memory as the controlled prefix. Otherwise the output labels its synthetic prefix. A missing EOS criterion is an explicit uncovered case. The result flags policy disagreement separately from successful execution and cannot establish which tokens .5232 emitted.

## Future zero-GRES worker

Use the originally qualified .5225 DTK image and Swift/helper/vision binds, no device attachment, privacy-preserving private scratch and mode-0700 output parent. The same exposed-image CPU load had one BF16 CPU model and requires 6 CPU and 64 GiB. The prior two-forward qualification took 496 seconds including model load, so allow the original outer reserve beyond a 520-second worker time and independent terminal readback. Root must separately authorize the actual numbered zero-GRES scope. All three visibility variables must be -1 and the original offline flags set to 1.

```sh
/usr/bin/python3 -B /stopping/run_cpu.py \
  --boundary-source /original-5225-boundary \
  --native-source /original-native-v4 \
  --original /approved/specialized-llava-val16-v4-5232-two-arms-all-cap-and-clean.json \
  --diagnostic-image /exposed/CLEVR_train_014344.png \
  --diagnostic-image-sha e0e9e1c96951283e0ce9db9bf220499301a6d7b147dd1221793157b8b8f0912b \
  --model /llava-base/model --model-receipt /llava-base/receipt.json \
  --output /diag-output/once --request "$request" --budget 520
```

The existing frozen diagnostic source directory contains cpu_probe.py and generation_boundary.py. The existing .5232 native v4 staging directory contains native.py, contract.py, repair.py and two_forward.py. Both mounts must be read-only. The production installed HF stopping provider and Swift template files and the three original model/tokenizer JSON declarations are pinned to authenticated readback request 24a9a75da04942d0a79885f9a7dc872a, SHA cb12278fd957a7898f7bca0915d055ac2aab32bec39c99048a008cb52d823abc. Base-arm configuration is exercised; CP25 runtime inference is outside this diagnostic.

Run python -B test_local.py. It verifies actual source and .5232/provider receipts, AST-equivalent request configuration, and synthetic controlled criteria plus the pre-forward sentinel. The local host does not load the model or execute a target Torch provider.
