# SDXL CP25 three-family paired product pilot

The fixed plan is `PLAN.json`, SHA256 `b4287b48cff81e86c740d446fa6314514f4102bad86f323dee6d83d679299dc4`. It covers cassette player, vintage electric kettle and vintage telephone-shaped wall clock. Each independent CC0 family has one fixed prompt and two unique seeds. The archived R33 terminal and public family admission receipts bind provenance. The training tool chest and painted wooden stool are excluded. No reference image feeds the GPU.

## Static stage with zero GRES

```sh
/usr/bin/python3 -B /pair-source/native.py verify-plan \
   --plan /pair-source/PLAN.json --r33-proof /approved/r33-terminal.json
/usr/bin/python3 -B /pair-source/native.py run --help
```

These commands validate the frozen plan and the actual .2675/.2676 archive without importing model providers or initializing a GPU. The latter is the exact archived `specialized-sdxl-2675-native-full25-cp20-exact-recovery-and-checkpoints.json`.

## Future fresh single-DCU owner

Use the original `R/containers/dtk26-sandbox-r1`, original explicit readonly `/training-runtime` mount, and `PYTHONPATH=/training-runtime/site` set after sourcing DTK. Preserve the existing ROCm and hyhal binding. Use offline Hugging Face and Transformers flags and `PYTHONDONTWRITEBYTECODE=1`. Bind a request-private mode-0700 `/tmp` bearing the exact `.sdxl-owner-request` marker. Bind a durable output parent owned by the worker UID with mode 0700. Preserve fresh physical admission, one numbered GPU step, and independent post in the external owner.

The read-only authority JSON has exact fields `request`, `step`, `node`, `boot`, `pci` with eight sorted PCI addresses, `epoch`, `deadline`, `GPU_GO` equal to true, `gpu_count` equal to 1, `plan_SHA256`, `output`, and `CP25_SHA256`. Its external SHA binds the exact scope. The worker compares actual numbered cgroup, boot, one visible HIP ordinal and PCI.

```sh
/usr/bin/python3 -B /pair-source/native.py run --request "$request" \
   --authority /approved/OWNER.json --authority-sha "$authority_sha" \
   --plan /pair-source/PLAN.json --r33-proof /approved/r33-terminal.json \
   --pipeline /official-sdxl --pipeline-manifest /first-update-source/pipeline.json \
   --asset-source /first-update-source \
   --adapter /r33/continuous/checkpoint-25/pytorch_lora_weights.safetensors \
   --output /durable/pair-once
```

The pipeline closes over the original real .5007 eighteen-file root and its exact manifest. `/first-update-source` is its original sibling source directory. Bind the actual R33 continuous checkpoint as a read-only subtree. The worker accepts only its 11,778,368-byte pinned adapter, checks the single active UNet adapter and compares loaded named values to the source. The original Euler scheduler and fixed 256×256, 30 denoising steps, CFG 5.0, blank negative prompt, BF16 UNet and text towers, and direct FP32 VAE loading apply to both arms. It materializes and verifies six BF16 latent bytefiles before any images. Each arm receives the same paired latent bytes and seed reset and records native prepared latent and realized timestep digests.

The worker writes `generations.jsonl` plus `result.json`. Every run retains the 12-slot denominator and six planned pairs. A hard interruption can leave durable per-slot JSON and will require owner reconciliation. Generated PNGs and latent files remain private.

## Independent assessment

Hide arm identity for the primary assessment. A slot passes only when it visibly contains one complete target object and a plain, clean background. Record both decisions and count missing or failed slots as unsuccessful. Report each family's paired difference with equal family weight. Keep incomplete pairs in the denominator.

The separate `score_cpu.py` supports the pinned official OpenAI ViT-B/32 provider manifest and weights. It requires all accelerator visibility variables set to `-1` and an independently authorized terminal document with `request`, exact `numbered_gpu_step`, `worker_result_SHA256`, and `independent_post_clean` true. Pass the document using `--terminal` and `--terminal-sha`, the same `--plan`, the sealed private `--generation` directory, `--weights`, `--provider` and a new `--output`. The scorer uses normalized image-text cosine only for complete equal-noise pairs and records all unfinished slots. Formal FID and general retention are separate.
