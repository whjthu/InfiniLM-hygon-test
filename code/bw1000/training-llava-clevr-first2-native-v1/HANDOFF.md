# T04 LLaVA CLEVR native first two updates

This candidate enters the deployed Swift SFT script. It overlays the original T04 model, dataset, output and callback paths. The eager, no-checkpointing BF16 recipe remains unchanged, including 4096 tokens, eight ranks, batch one, accumulation four, four DataLoader workers, 25 scheduler steps and AdamW cosine. The worker stops through the native callback after the second optimizer boundary and requests the native CP2 save.

Use the original DTK sandbox and the read-only runtime bindings from the actual .4860 runner. Mount the original recovered 18-file model read-only. Its target receipt and each member's extent, inode and mtime are checked before Swift opens the model. Model compatibility is imported once through the qualified OneVision module and the temporary cache alias from .4860. The normal Swift model loader then runs in the rank's actual HIP placement. The native checkpoint finalizer must report zero discrepancies and the loaded class must match the preimported class.

## Mount contract

| Guest path | Source |
| --- | --- |
| `/llava-source` | These source files |
| `/llava-train-only` | Exact original .4719 train JSONL and only the 64 PNGs referenced by it, as a read-only view |
| `/llava-framework` | Original read-only 393-file framework from the .4719 deployment |
| `/swift-runtime` | Original approved runtime from .4719 and .4860 |
| `/vision` | Original seven-member vision provider from .4860 |
| `/llava-helpers` | Read-only closure below |
| `/llava-run` | New exclusive request-owned 0700 output directory |

The helper root contains the pinned original files `probe.py` from `training-llava-clevr-native64-cpu-v1`, `llava_recovered_native_cpu.py` and `llava_target_eager_import.py` from tools, and `cache_alias.py` from `training-llava-target-native64-meta-CPU-v4`. The latter eager-import helper is qualified by the paired target .4896 zero-GRES import check following the .4890 eight-rank failure. The remaining helper names are `VISION_FILES.json`, `FRAMEWORK_FILES.json`, `NATIVE_BINDING.json` from the exact V4 sources, `MODEL_RECEIPT.json` from the recovered target model, `original-launch.json` from `training-llava-A100-first-updates-v1`, the original .4719 `INPUTS.json`, aggregate public `4860.json` and `4719.json`, and privately generated `TENSORS.json`.

Register the input observer before Swift's multimodal pre-hook. The original .4901 run verified that all eight ranks entered the training loop with complete collated inputs. Swift then converted `input_ids` to `inputs_embeds` before the previous observer ran, and all ranks failed at that observer. Prepending the observer keeps the input fingerprint at the collated boundary and lets Swift apply its original transformation before the model call. A completed forward still requires a corresponding finite native loss record.

`TENSORS.json` must come from an independently successful target zero-GRES diagnostic on the same original training inputs. Its fields are `binding_sha256`, `train_sha256` and `rows` with `ordinal`, `row_sha256`, `image_sha256`, `input_sha256` per row. The native fingerprint includes the four collated inputs `input_ids`, `labels`, `attention_mask`, `pixel_values`. Integer casts apply only to the first three. Pixel dtype, shape and list/tuple identity remain exact. The worker checks the prepared input and actual model forward separately. If a provider changes those tensors between the two boundaries, the rank preserves the type/shape evidence before refusing the forward.

## Inner launch

The root controller must first acquire actual queue custody and independently establish an eligible eight-DCU node, a new label, request, port and output, and clean same-boot physical PCI/KFD/fd/port checks. An actual numbered step owns all eight devices. In that step the root authorizes once with an exclusive `/llava-run/OWNER.json` and supplies its SHA as `LLAVA_OWNER_SHA256`. Required fields are `request`, actual numeric `step`, `node`, `epoch`, `deadline`, `gpu_count:8`, `port`, `fingerprint_sha256` and `GPU_GO:true`. Use a bounded epoch of at most 1500 seconds. Allocate adequate CPU headroom for four native workers per rank. The zero-GRES .4860 memory and runtime figures do not establish eight-card training fit.

Run within the original DTK environment, with fresh request-private local HOME, caches, TorchElastic logs and TMP outside the read-only trees. This command is the only business child of the bounded supervisor.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec /usr/bin/python3 -B -m torch.distributed.run --nnodes 1 --nproc-per-node 8 \
  --master-addr 127.0.0.1 --master-port "$PORT" --log-dir "$REQUEST_LOG" \
  /llava-source/train.py --run /llava-run --model "$MODEL_RO" \
  --request "$REQUEST" --fingerprint-sha256 "$TENSORS_SHA"
```

Use an already qualified owned-child supervisor to run the one process group, bound the whole epoch and terminate only that process group if cancelled or timed out. Independently settle actual scheduler accounting and run a zero-GRES post step to verify reaped children and renewed boot, assigned PCI, KFD/fd and port clearance. Preserve partial rank/optimizer receipts and any directory of unknown ownership. A launch receipt or a loss alone does not imply successful updates or release.

After confirmed terminal status, run `python -B /llava-source/verify.py /llava-run "$REQUEST"` independently. Combine the rank and file result with accounting, physical post and the immutable source/input receipt. A successful artifact must contain all 64 distinct train-only forwards, all eight ranks at both real optimizer boundaries, 504 AdamW counters per rank, unchanged weights at zero LR, a common nonzero change at positive LR, and the complete native CP2 inventory. Carry the 64-forward and 16-rank-update denominators on any failure. This source neither launches Slurm nor creates an execution authorization.
