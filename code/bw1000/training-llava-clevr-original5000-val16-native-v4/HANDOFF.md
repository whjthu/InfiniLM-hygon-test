# Original LLaVA .5000 CP25 on independent CLEVR val16, native worker v3

The diagnostic scope `.5203` identified 698 original GPU BF16 base tensors, zero LoRA tensors and no PEFT configuration. The previous 1202 count was taken in `.5006` after the CPU loader injected its 504 LoRA tensors. The `.5203/.5204` archive is pinned at `3d18fb802f27471014c03406e8688d49c698db02ef386d4ce3637b97736da79a`. v3 requires 698 base tensors, zero LoRA, and exactly one original Transformers checkpoint finalizer with zero missing, unexpected and mismatched keys. The same pinned CPU loader's observer wraps only the one GPU model-loading call.

The sealed result uses `facts.base_diagnostic` when the base gate passes and `failure.base_diagnostic` when loading or its gate fails. It records 13 predicates and the bounded device/dtype, LoRA and PEFT fields. The new predicate keys are `parameter_tensors_698`, `zero_lora_parameters`, `finalizer_called_once`, `finalizer_missing_keys_zero`, `finalizer_unexpected_keys_zero` and `finalizer_mismatched_keys_zero`. The last four are supported by `checkpoint_finalizer` with `calls`, `missing_keys`, `unexpected_keys` and `mismatched_keys`. A missing or ambiguous count is null and fails its predicate. Parameter names, provider key examples and PEFT values remain private. A fresh owner must bind the v3 manifest.

The worker loads the original OneVision base once, then uses the pinned Swift engine to run the bare base followed by the original CP25. It checks all 504 active named adapter tensors against the stored checkpoint at the first CP25 generation boundary. For each slot it compares the actual collated prompt IDs, image pixels and available image metadata between the two arms, then checks greedy decoding with a 128-token cap.

Mount the previously qualified .5006 assets and helpers read-only, with the independent answerless input mounted at `/llava-blind` and no gold mounted. The original model is `R/results/training-llava-original-model18-v1`. The original training root is `R/results/training-llava-clevr-resume25-0e4a27af103c`. Reuse the `/llava-helpers`, `/llava-framework`, `/swift-runtime` and `/vision` layout from `tools/run_llava_clevr_cp25_reload_target.py`, and the same DTK image. Bind the original `tools/llava_clevr_cp25_reload.py` as `/scope/llava_clevr_cp25_reload.py` and check its embedded SHA. The low-level runtime requires the exact Torch, Transformers 5.5.0, and Swift 4.4.2 providers. Keep the newly assigned HOME, MIOpen directories, HF cache and tmp writable only inside the unique scope. Source `/opt/dtk/env.sh`, preserve the exact proven ROCm bind and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1` in the container. The worker creates its own request-scoped HF dynamic-module cache under private `/tmp`.

The independent resource owner must check ordinary fleet exclusion, exact single-DCU numbered step, current boot and the eight permitted PCI addresses, and independently reconcile the terminal and physical return. Write an exclusive, read-only owner document only after identifying that step. Use exact keys `request`, `step`, `node`, `epoch`, `deadline`, `gpu_count` equal to 1, `GPU_GO` equal to true, `boot`, `pci`, `requests_SHA256`, `images_SHA256`, `output`, and `checkpoint_SHA256` equal to `b8b8a8b9dfe5012c4ce5f80499a1b3fc5e7b8b140a7a734cada07c4a39b83719`. The output child must be absent and its parent must be owned by the current UID with mode 0700. The guest will independently bind the Slurm IDs, cgroup, boot, single visible HIP device, and selected PCI.

```sh
/usr/bin/python3 -B /scope/native.py run \
    --input /llava-blind --requests-sha "$requests_sha" --images-sha "$images_sha" \
    --owner /owner/OWNER.json --owner-sha "$owner_sha" --request "$request" \
    --output /eval-output/once --model /llava-base/model --model-receipt /llava-base/receipt.json \
    --resume-receipt /llava-trained/independent-resume25.json \
    --checkpoint /llava-trained/train/checkpoint-25 --reload-tool /scope/llava_clevr_cp25_reload.py
```

The `verify-input` mode takes only the same three input arguments. It verifies the actual set of 16 PNGs, exact requests and image manifest without importing model providers. The sealed `predictions.jsonl` has 32 unique rows in base then CP25 order, each with `arm`, `slot`, `outcome` and `text`. Outcomes are `eos`, `cap`, `stop`, `error` and `missing`. The first actual error keeps its slot and seals outstanding slots as missing. `result.json` stores the ledger SHA and byte length, the outcome counts and the two-arm evidence. Independently reconcile Slurm and the ledger before scoring. An external kill that prevents sealing requires an independent missing-slot accounting using the durable per-slot files. No answer keys, scoring, optimizer updates, or formal delivery credit are part of this worker.

## Scoped cache repair

Val16 v4 carries the exact repair.py and two_forward.py used by the original
.5226 exposed-image zero-GRES qualification. The latter supplies only shape
and cache helpers to the patch. No forward stop, exposed input, logits or
diagnostic token witness is part of Val16. The patch is installed on the
exact OneVision instance once for both arms, before the first infer call or
CP25 adapter construction. Its pinned installed source checks remain in
force. It preserves native positions already supplied by Transformers and
uses the native DynamicCache visual decision for the validated None branch.
Each successful slot records the patch's entry and visual-prefill/decode
counts as per-arm aggregates only. Gold remains unmounted.

The original .5226 receipt SHA-256 is
7e4bf2dcb4e5cf35ae306a842177ed8f8c3d26aaff1f4327338331de0e44e83a.
The real two CPU forwards establish that narrow CPU route. GPU and active
adapter integration still require the new owner run.
