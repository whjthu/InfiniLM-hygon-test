# SDXL four-row CP25 CLIP diagnostic

The scorer binds the actual .2821 numbered generation and clean .2822 post, the independently sealed three-arm index, the original .2762 two-arm generation ledgers and exact .2765 CPU scores. It validates arm, slot, family, seed, prompt and all three noise/timestep hashes for each comparison. The old 12 PNGs are already verified by the independent composition. This scorer reads their original pinned ledger metadata and reuses their 12 actual .2765 cosine values. It never opens the old image files or reencodes them.

Only the six four-row CP25 images are read, rehashed and encoded. The scorer uses the exact .2765 official OpenAI ViT-B/32 provider, 353,976,522-byte weight file, native clip.load JIT false and its returned image preprocessing transform. Text is encoded as the same six prompts. New images use two per native batch when both are available, preserving complete-run image batch size. The report gives four-row minus two-row CP25 as the first contrast and four-row minus base as the second. It keeps all six planned pairs and reports counts, per-family means and the equal-family means when every family has coverage. Image encodings, scored pairs, errors and missing slots are separate counts.

The report contains no human rubric, FID, formal training credit or domain delivery credit.

## Target zero-GRES invocation

Reuse the actual .2765 n03 DTK CPU layout, same official CLIP runtime provider tree and ViT-B/32 weight, and 4 CPU/8 GiB/0 GRES. The root must separately authorize the actual numbered step and check terminal accounting. Use a fresh private mode-0700 output parent. All inputs are read-only.

```sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B /score-scope/score_cpu.py \
  --plan /approved/PLAN.json \
  --terminal /approved/specialized-sdxl-2821-four-row-cp25-six-images-and-three-arm-index.json \
  --composition /three-arm \
  --original /original-2762 \
  --new /new-2821 \
  --prior-report /approved/specialized-sdxl-cp25-2765-three-cc0-families-clip-score-report.json \
  --prior-scorer /approved/previous-clip-score-v2.py \
  --weights /clip-weights/ViT-B-32.pt --provider /clip-provider \
  --output /private-output/result.json --request "$request"
```

The read-only original aliases are .2762 results/sdxl-cc0-cp25-pairs-fp32-94f0acaeec61/durable/pair-once and .2821 results/sdxl-cc0-four-cp25-six-ev2-a6b938dcee0e/durable/four-row-pair-once. The independent composition directory must contain result.json and three-arm-index.private.jsonl at SHA-256 3648cc4458711f150d7a6a843b6478e497e4a0c4d110255ee5c5083a2b575289 and 0cc85b1e5e3181c4dcf003cd250e6f8eddb4b7c3fb4935285d28cea41cd19eeb. Use the previous source code/bw1000/training-sdxl-cc0-cp25-clip-score-v2/score_cpu.py (59feca66365c73431a963ad8500bba5c0a42a4579139d9ee40f8bc5663552772). The provider manifest SHA-256 is 77202606488902993d5fea75b321f1e9348298112e6ce42f40d52d5df7471196. The actual weight SHA-256 is 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af. These are the original .2765 artifacts, with no download needed.

Run python -B test_local.py for original archive, plan and scorer identity checks and wrong slot, seed, timetable and denominator regressions. Local verification does not open generation PNGs or import the CLIP provider.
