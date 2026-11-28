# SDXL three-arm follow-up, four-row CP25 only

This worker generates six new outputs from the actual `1194900.2769` continuous CP25. It uses the original six-slot plan and the already-verified Diffusers/Euler/FP32 VAE pipeline. It reads the six original BF16 latent bytefiles and the .2762 ledger in place. The original effective Euler configuration and each slot’s prepared latents and actual timestep vector must reproduce their original SHA-256 values. Native 256×256, CFG 5, 30 steps, empty negative prompt, fixed prompts and seeds, original BF16 UNet/text towers and FP32 decode are unchanged.

The original .2762 base and two-row CP25 images are never generated again. There is one new pipeline load and one four-row CP25 adapter loaded and checked at 1.0 strength with the original 1,120 named values. The continuous CP25 LoRA is 11,778,368 bytes with SHA-256 `7301c4f968e79b93c1ff183b03f2275789d61de06565d8b2c2ea091af27b876a`. Its native checkpoint manifest SHA-256 is `b108ecd38a9ca05ffef58ce5454adc3db9169282d2a2f79a8a057613c4bbe14a`. The manifest binds four ordered training rows, the native full-state boundary and the on-disk adapter member. That actual readback was observed in request `0d0eb974acd54b2b9628660b990f8c8a`. The original plan’s two-row checkpoint pin is preserved for historical base/control identity.

## Original mounts and target commands

Use the original DTK image and offline settings from `training-sdxl-cc0-cp25-three-family-native-fp32-vae-v2`. The root owner must provide fresh exact single-DCU authority and independent zero-GRES pre/post on `1194900/f13r4n03`. The original native guest authority schema is unchanged, including `CP25_SHA256` which takes the new 7301c4f9 digest. The private scratch marker and durable output parent must belong to this distinct request.

Read-only mounts:

- Source files and plan at `/pair-source`, existing official pipeline at `/official-sdxl` and its exact sibling `source` at `/first-update-source`.
- `R/results/sdxl-cc0-cp25-pairs-fp32-94f0acaeec61/durable/pair-once` at `/original-2762`.
- `R/results/sdxl-cc0-four-full25-cp20-a9a91b5727d8/durable/once/continuous` at `/four-row-a9a91b5727d8/continuous`. Only the native manifest and adapter at `checkpoint-25` are used.
- The old .2675 terminal archive at `/approved/r33-terminal.json` and actual .2769 terminal archive at `/approved/four-row-terminal.json`.

From the approved guest shell, reuse the following positional shell arguments for either native command.

```sh
CP=/four-row-a9a91b5727d8/continuous/checkpoint-25
set -- --plan /pair-source/PLAN.json --r33-proof /approved/r33-terminal.json \
  --training-proof /approved/four-row-terminal.json --original /original-2762 \
  --checkpoint-manifest "$CP/genmlu-checkpoint.json" --adapter "$CP/pytorch_lora_weights.safetensors" \
  --adapter-sha 7301c4f968e79b93c1ff183b03f2275789d61de06565d8b2c2ea091af27b876a \
  --checkpoint-manifest-sha b108ecd38a9ca05ffef58ce5454adc3db9169282d2a2f79a8a057613c4bbe14a

/usr/bin/python3 -B /pair-source/native.py verify-plan "$@"
```

The verification reads the original sealed ledger and six noise files and checks the .2769 manifest and adapter bytes. It initializes no model or accelerator. After separate authorization, the native GPU command is:

```sh
/usr/bin/python3 -B /pair-source/native.py run "$@" --request "$request" \
  --authority /approved/OWNER.json --authority-sha "$authority_sha" \
  --pipeline /official-sdxl --pipeline-manifest /first-update-source/pipeline.json \
  --asset-source /first-update-source --output /durable/four-row-pair-once
```

## Independent three-arm assembly

After reading actual numbered worker accounting and performing the independent physical post, prepare the existing normalized terminal certificate shape with exactly `request`, `numbered_gpu_step`, `worker_result_SHA256`, and `independent_post_clean`. Use its SHA.

```sh
/usr/bin/python3 -B /pair-source/comparison.py --plan /pair-source/PLAN.json \
  --original /original-2762 --new /durable/four-row-pair-once \
  --terminal /approved/new-terminal.json --terminal-sha "$terminal_sha" \
  --output /private/new-three-arm-index
```

The composer verifies all 12 original PNGs and each available new PNG against the two sealed ledgers, keeps all 18 planned slots, checks the direct comparison between two-row `cp25` and `four_row_cp25` and writes only a private index and aggregate summary. It leaves the existing anonymous review untouched and does not produce ratings, FID or delivery credit. Any missing new generation stays missing. A completed PNG with a mismatching noise or timetable digest fails closed.

Local verification is `python -B test_local.py`. It covers the actual published plan and .2769 pin readback, fixture CP25 contamination rejection, file byte tamper, 18-slot metadata composition, and a timestep mismatch. It uses fixture PNGs and makes no model or device calls.
