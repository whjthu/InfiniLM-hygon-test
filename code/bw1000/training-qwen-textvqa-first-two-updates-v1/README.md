# Qwen3-VL street-scene first two updates

This candidate reuses the native input stage from CPU step 1194900.1251. The 41 original records and 16 images are opened at the existing read-only destination. The original worker report binds each record to exact token and label digests and image grid. Each actual microbatch must identify one unique original record through those values before the original training step runs. The working directory is the input directory so the original relative media names remain unchanged.

The command inherits the .220 training recipe, model inventory of 14 files, and original LoRA initialization. It retains eight BF16 ranks, per-rank batch 1, accumulation 4, LoRA 16/32/0.05, frozen vision, length 4096, AdamW with LR 1e-4, and a nominal 25-update cosine schedule. The observer requests a stop after the second attempted update, disables periodic saves and leaves all results private. It records actual AdamW counter transitions, skip status, before/after trainable digests, losses and per-rank epoch and padding records. The first qualifying nonzero update requires a parameter change and an AdamW counter advance on all eight ranks. The nominal warmup can give an update with a zero learning rate without changing parameters.

The configured global batch is 32. The local 41-row fixture with the stock shard geometry produced batches of 32 and 16, including 7 repeated sampler-padding slots. The run-time guard verifies the actual sampler type and geometry. Independent aggregation checks the full permutation and repeated prefix against observed rows. The local runtime lacks Torch, Accelerate, Transformers and Swift. Use `check_runtime_sampler.py` in the approved CPU environment for a stock-class-only check. The actual Trainer still has to report its own observed geometry and batches.

From an independently admitted original eight-DCU compute step, place the source files together under a verified absolute path. Pass the current exact assignment produced by the existing original eight-device gate, the existing model-source and platform pins and the mounted data root. The outer owner retains placement, model SHA verification, scheduler accounting and cleanup ownership.

```sh
python3 -B "$SOURCE/native_entry.py" \
  --request "$REQUEST" --parent "$PARENT" --node "$NODE" \
  --output "$RUN" --assignment "$RUN/assigned.json" --assignment-sha "$ASSIGNMENT_SHA" \
  --framework "$FRAMEWORK" --original "$ORIGINAL_LAUNCH" --binding "$ORIGINAL_BINDING" \
  --recipe "$ORIGINAL_RECIPE" --owned "$ORIGINAL_OWNED_PROCESSES" \
  --public-native "$NATIVE_1251_PUBLIC_RECEIPT" --terminal220 "$ORIGINAL_220_TERMINAL" --assets "$ORIGINAL_220_ASSETS" \
  --port "$PORT" --epoch "$EPOCH" --deadline "$DEADLINE" --maximum "$MAXIMUM"
```

The pre-created private output directory is `results/qwen-textvqa-first-two-$REQUEST` in the original project. Native intent, one original child process forest, partial rank receipts, and its outcome are exclusive writes. A failed or interrupted run retains the planned denominator of two updates. Results remain private until independently evaluated.

Run `python -B -X utf8 check_local.py` from this directory to check real source and data identities and capture fixtures.
