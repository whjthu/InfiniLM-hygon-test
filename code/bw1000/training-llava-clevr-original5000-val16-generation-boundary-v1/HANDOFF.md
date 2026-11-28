# LLaVA .5207 bounded generation diagnosis

The frozen v3 worker remains unchanged. The original .5207 failure receipt records a TypeError SHA256 `36c357cc64e91c70ef5ed3a22e2348665c772c896c4ce9fd8e48c2a0568ddba7`, corresponding to `'NoneType' object is not subscriptable`. Original .5208 physical return passed. The diagnostic compares actual exception bytes to that digest. A matching message alone cannot identify the cause or establish CPU/GPU equivalence.

The observer wraps the real native template `generate` and `get_generate_ids` and the actual model's `generate`, `prepare_inputs_for_generation` and first `forward`. It keeps the original signatures through `functools.wraps` and restores instance descriptors. It reports entry, return, exception, callable origin file SHA, fixed method labels, allowlisted keyword presence/types and bounded traceback module/function/line projections. It reports a None operand only when the deepest supported traceback frame executed binary subscription, the source line contains one direct named subscript operand, and that local actually held None. Ambiguous or unproven operands are null.

## Dedicated zero-GRES qualification

Reuse the original .5006 read-only OneVision model, recovered loader helpers, canonical 393 framework files, Swift runtime, vision and DTK mounts. Use no ROCm attach, set all three accelerator visibility variables to `-1`, and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1` before Python starts. Use a request-owned private `/tmp` and mode-0700 output parent. The original .5006 CPU bare-base provider took about 283 seconds to load. The candidate has a 420-second default deadline and a bounded range of 60–480 seconds.

```sh
/usr/bin/python3 -B /diag/cpu_probe.py \
   --native-source /original-native-v3 \
   --diagnostic-image /exposed/CLEVR_train_014344.png \
   --diagnostic-image-sha e0e9e1c96951283e0ce9db9bf220499301a6d7b147dd1221793157b8b8f0912b \
   --model /llava-base/model --model-receipt /llava-base/receipt.json \
   --output /diag-output/once --request "$request" --budget 420
```

Use the exact 193,647-byte official train image from the prior exposed train-only staging path. Mount the single PNG, with its original basename. The CPU invocation uses the same OneVision prompt template and greedy MaxTokens128 configuration. It changes only device placement to CPU and exits at the pre-forward sentinel or an actual earlier exception. It does not open Val16, adapter or answer files. It writes a diagnostic receipt without a blind prediction ledger. Test code verified the actual exposed PNG and its original training index.

The independent `derive_native.py --native-v3 … --out …` command can render an optional first-base-slot witness for a later approved native run with the same v3 inference and ledger behavior. It requires the exact frozen v3 digest, compiles the derived worker and refuses existing output. It supplies no new GPU authorization. The zero-GRES qualification is required before considering such a scope.
