# SDXL Euler config wire isolation, zero GRES

This diagnostic loads just the official scheduler configuration through native Diffusers 0.39 on the original Hygon DTK image. Two separate Python processes use PYTHONHASHSEED 17 and 101. A native EulerDiscreteScheduler.from_pretrained, its initial sigma tensor, and native set_timesteps(30, device="cpu") are observed independently in each process.

The parent reuses the byte-for-byte serialization used by the original GPU worker. It compares every configuration field and separately tests normalization of only the order in _use_default_values. The old full raw wire digest is searched by permuting only that one list with its exact observed members and preserving all other fields, up to 100,000 candidates and 6 seconds for each distinct observed normalized config. A bounded search without a match is distinguished from an exhaustive search. A raw wire match and observed CPU timeline agreement are separate facts.

It compares actual tensor dtype, shape and bytes for timesteps and both sigma observations. Timesteps are independently checked against all six original paired .2762 digests recovered from the exact staged result and ledger. The original run did not retain sigma digests. The two new CPU sigma observations therefore have no historical reference. This probe cannot establish the cause of .2774 and does not change the GPU worker.

## Future target command

Use the original n03 DTK image, exact read-only training runtime at /training-runtime with PYTHONPATH=/training-runtime/site, a fresh mode-0700 result parent and private /tmp. The authorized target must allocate zero GRES with all three device visibility variables set to -1. Use the original offline settings. Mount only the official SDXL scheduler directory from the verified production 18-file pipeline, the sealed .2762 pair-once directory and these sources read-only. The worker reads only result.json and generations.jsonl from .2762. It does not open its images.

```sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/training-runtime/site
/usr/bin/python3 -B /euler-scope/euler.py run \
  --scheduler /official-sdxl/scheduler \
  --original /original-2762 \
  --output /diag-output/once --request "$request"
```

The source pins include scheduler_config.json af3e45a9…, configuration_utils.py 4a7af9be…, Euler implementation ecee4a11… and the old effective wire b9e0375a…. No model import/initialization or image generation occurs. The supervisor runs the two children sequentially with at most 35 seconds each, bounds reconstruction, and writes a private result plus per-child returncode and log digests. Produce separate original numbered accounting and reconcile without replay before interpretation.

Run python -B test_local.py to check the real archived provider/wire projection, actual independent Python process seeds, a fixture six-pair ledger and targeted counterexamples for altered non-default fields, wire search exhaustion and altered timeline bytes. This local test does not import the production Torch or Diffusers provider.
