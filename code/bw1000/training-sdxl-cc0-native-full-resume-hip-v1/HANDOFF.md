# SDXL CC0 native HIP full checkpoint and recovery

Use a fresh scope with the original .2670 DTK runtime mounts and a real single-DCU grant. The source renderer verifies the exact original trainer, common gencuda backend and native checkpoint helper before it derives the isolated HIP entry. The original MLU helper remains frozen. The HIP checkpoint schema requires the actual Torch HIP CUDA API, one device ordinal zero, a private generator clone round trip and the original checked Accelerate CUDA RNG producer inventory. The .2670 checkpoint supplies independent evidence for the serialized RNG keys and 16-byte device RNG storage. Its checkpoint-1 is excluded as a resume source.

Both arms use the same verified two-input ImageFolder and recipe. Workers are zero for full checkpoint mode, with BF16, batch one, accumulation one, LoRA rank and alpha four, AdamW at constant 0.0001, and 25 nominal steps. The continuous native process saves at steps 10, 20 and 25. The second process loads the explicitly selected continuous CP10 or CP20, restores the original native sampler and full state, and runs through step 25. CP10 yields 40 updates in total, and CP20 yields 30. The original trainer has no manual skip in its full-checkpoint branch. Comparison requires independent native process identities, exact pre-update restored state, all paired sample witnesses, finite losses and gradients, exact paired values and exact semantic final state. The report preserves the first divergent update if observed.

## Qualified target inputs

Pass the .5007 read-only official pipeline and its original `source/pipeline.json`, .5039's four exact raw input members, the original .5099 framework source root containing both `gencuda` and `genmlu`, and the original `b3e85543…` trainer file. Mount the two frozen first-update sources from .5007 at `/first-update-source` for reuse of the existing asset and ImageFolder checks. Keep the original `/training-runtime/site` as the explicit Python path. The owner must supply the independently identified DTK image digest, a fresh dedicated mode-0700 `/tmp` with the same `.sdxl-owner-request` marker as .2670, and a new persistent result directory. The source-only package leaves scheduling, component mounts and independent physical pre/post to the owner.

```sh
SDXL_SINGLE_DCU_OWNER_REQUEST="$request" PYTHONPATH=/training-runtime/site \
  /usr/bin/python3 -B /hip-resume/run.py run --request "$request" --image-id "$image_id" \
  --resume-step 10 --firstupdate-source /first-update-source \
  --original-trainer /stock-trainer.py --framework /framework \
  --pipeline /official-sdxl --pipeline-manifest /first-update-source/pipeline.json \
  --input /two-cc0 --work /durable/paired
```

Use `--resume-step 20` for the shorter explicitly authorized plan. `verify-source --trainer … --backend … --helper …` derives and hashes all three files without model imports. `test_local.py` exercises the exact three original source bytes, the derived actual backend command and trainer parser, the HIP device-class gate with a fixture, the archived real static RNG receipt, and both comparison denominations. No local GPU calls or numerical continuity claims are part of this source validation.
