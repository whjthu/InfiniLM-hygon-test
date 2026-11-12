# MLU Training

Cambricon MLU adapters for multimodal language and generative models.

## Layout

- `vlm/`: ms-swift training, native TextVQA subset validation and compatibility plugins.
- `generation/`: inference, training and reinforcement-learning backends, grouped by model.
- `tools/`: repository checks shared by both components.

Each component documents its dependencies and commands in its own README. Model
weights and datasets are external inputs. Use an external output directory for
checkpoints, generated media and metrics. Shared weights under `/mnt/models` are
read-only inputs.

## Hygon training recipe entry

Use the unified entry to list the 17 frozen A100 recipe IDs:

```sh
python tools/hygon_training_recipe.py list
```

For a generation or geometry case, list its original assets before binding target paths:

```sh
python tools/hygon_training_recipe.py T09.dp2 --a100-launch /references/hunyuan/launch.json
```

Pass `--asset-map /path/to/assets.json` with an exact mapping from each original
absolute path to its Hygon path. The output preserves the original training
configuration, including the Hunyuan DP2 world size and SP setting. For a VLM
case, use the same entry with `--help` after the case ID to see its Swift path
arguments.

For VLM training, run the launcher with the selected Swift environment's
Python in a single-node Slurm step with eight visible cards. For example:

```sh
/opt/swift/bin/python tools/hygon_training_recipe.py T01.sft \
  --a100-launch /references/qwen/launch.json --swift /opt/swift/bin/swift \
  --models /models --dataset /data/sft.jsonl --plugins /sources/vlm-plugins \
  --output /experiments/qwen-first --master-port 29501 --run \
  --receipt /experiments/qwen-first.launch.jsonl
```

The receipt and training output must use new paths under existing parent
directories. The launcher checks HIP and eight visible cards with the
same Python interpreter and environment used to start Swift. Add
`--dry-run` to record the plan before inspecting hardware or starting Swift.

For controlled-input VLM diagnostics, supply both
`--diagnostic-plugin /sources/capture.py` and `--diagnostic-callback capture_updates`.
The entry keeps the frozen training arguments and native callback, appends the
observer, and marks the resulting plan ineligible for formal performance claims.
The observer source and every external input still need their own frozen identities.

`vlm/runtime/dropout_capture.py` can capture the actual native dropout mask or
replay a SHA-bound mask while preserving the backend's native RNG advancement.
It observes the native sampler directly, so zero-valued inputs do not hide draws.
The record includes probability, shape, stride, dtype and the complete CPU boolean
mask. Replay requires those input properties to match. These are diagnostic
operations with additional copying and arithmetic; CPU checks do not qualify a
GPU backend or establish cross-platform numerical tolerances.

Generation and geometry cases also accept `--run` from an authorized allocation:

```sh
python tools/hygon_training_recipe.py T09.dp2 --a100-launch /references/hunyuan/launch.json \
  --asset-map /references/hunyuan/assets.json --run --cards 0,1 \
  --source-root /sources --output-root /experiments --run-name hunyuan-dp2-first \
  --lock-root /shared/card-locks --lock-host "$HOSTNAME"
```

Add `--dry-run` to write a planned manifest. The mapped Hunyuan checkout in
this example must be `/sources/hunyuanvideo-1.5`. VGGT also needs
`--vggt-weights` because its frozen recipe is a config file without a
weights field. SDXL DDPO uses its runtime package without an upstream checkout.

The launcher checks the recipe's card count and uses `gencuda --accelerator hip`
from this checkout. The runtime manifest records the reported platform and
version. A wrong platform or visible card count fails before training.
Target asset identity and training results are verified separately.

For SDXL SFT, the first-step replay uses the original A100 r2 reference
and its frozen equivalence policy:

```sh
python tools/hygon_sdxl_first_step.py --reference /references/sdxl/reference.pt \
  --policy /references/sdxl/policy.json --verify-reference
```

For the device replay, replace `--verify-reference` with
`--weights /models/sdxl-1.0 --output /experiments/sdxl-first-step` in an
isolated one-card HIP step. This records one UNet update with the frozen
realized inputs. Run the CPU comparison from the same checkout:

```sh
python tools/hygon_sdxl_compare_first_step.py --reference /references/sdxl/reference.pt \
  --policy /references/sdxl/policy.json --observed /experiments/sdxl-first-step/observed.pt \
  --receipt /experiments/sdxl-first-step/summary.json --output /experiments/sdxl-first-step/compare.json
```

The report includes per-tensor errors and exits nonzero when the comparison
fails. VAE, text encoding, recovery and performance require separate evidence.

## Scope

This is a source delivery for bounded MLU workflows, not a fully qualified
twelve-model training platform. The VLM component provides five SFT entry points
and GT-free native TextVQA subset generation with separate official scoring;
it does not provide VLM RL or a vLLM sidecar. The generation component provides
six executable inference backends, six component-training backends, and four
native RL backends. TRELLIS has a separate bounded 512 inference path: it can
CPU-decode shape/material fields and dual-grid geometry, and its independent
decoded-cache component can export an opaque textured GLB using verified
external CPU UV dependencies. End-to-end complete output additionally requires
the generation backend to preserve every native sparse-structure coordinate;
a coordinate-limited subset is not a complete image-to-3D result. This does not
imply unrestricted upstream pipeline parity or full-4B training.

An available command is not a guarantee of model quality, performance, or
recovery equivalence for every configuration. Those conclusions require
external evidence tied to the tested commit, runtime, weights, and workload.
See the component READMEs for training scope and unsupported interfaces.

The new owned TextVQA workflow has hardware qualification for Gemma base on
the main host only. Other model routes require their own deployment validation;
historical runs retain their original controller and source identities. This
64-question subset workflow does not establish full-benchmark quality, trained
adapter quality, performance or recovery acceptance.

## Training input identity checks

The normal `T08.flow-matching-lora` launch also verifies its pinned training
manifest before importing the device backend. This case requires 128 frames
and all 201 motion features; a manifest containing 128 samples at a shorter
frame length is a different workload. Resolve that recipe/reference mismatch
before training. This metadata check never reads cache tensor bodies or changes
the frozen model config, noise, targets or sample order. Dry runs remain plans
with unverified inputs.

`tools/hygon_vggt_input_check.py` checks already-staged original CO3D inputs
without loading model weights, extracting cached features or starting training.
Supply the original config, private reference manifest and upstream source
manifest with their SHA256 values, the pinned backend SHA256, the input and
upstream roots, a new private output path, and the owning CPU job's original
`--epoch` and `--deadline` (at most 240 seconds apart). Run with
`CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=` in an
externally bounded CPU-only step. Deadline callbacks detect exhaustion between
operations; the owning job must also bound blocking imports and native decoding.
This command does not stage files or authorize a dataset transfer.

The checker uses the unchanged native conversion, depth filter and tensor hash
functions. It repeats the native candidate order and compares rejected attempts
as well as every accepted batch's five tensor fields and the combined batch
identity. A mismatch fails the check while retaining the complete attempt
denominator. Decode/deadline failures preserve completed and planned counts in
the adjacent private `.progress.jsonl` file. No raw tensors are saved. Input
identity qualification is separate from model numerics, training, recovery,
performance and held-out task effects.

`tools/hygon_hymotion_meta_check.py` constructs only the native motion Transformer
on meta parameters and calls the unified training backend's original LoRA and
gradient-checkpointing helpers. Supply a SHA256-pinned metadata specification,
sixteen pinned upstream Python files and the owning externally bounded CPU
step's original epoch/deadline (at most 220 seconds apart). It compares the
complete trainable schema against an original A100 meta reference, checks the
frozen base and empty optimizer binding, and records actual runtime module
identities. Explicit empty device visibility is required. This does not read
weights or cache tensors, construct the full pipeline, qualify a frame length,
run training, or establish numerical/performance/effect acceptance.
The CPU checker excludes optional FlashAttention imports before native network
construction because a device extension may query GPU properties during import.
This is scoped to this meta checker, restores the import hook afterward, and
does not change the training backend or qualify production attention.


The fixed SDXL first-update replay and selected backward observer accept
`--accelerator cuda` for a source-A100 provenance check and default to `hip`
for the existing target route. They verify the actual runtime and exactly one
visible device before reading weights. The fixed model, input, loss, backward,
optimizer and capture body is shared; selecting CUDA does not change numerical
tolerances or establish a fresh reference. Observed tensors stay in the owning
experiment storage and require independent comparison with the original policy.


`tools/hygon_sdxl_operand_attribution.py` consumes the two SHA256-pinned original
selected-LoRA captures already staged in experiment storage. The shared operand
algebra accepts an explicit rank and parameter-gradient dtype, preserving the
existing rank-two/FP32 default while supporting this rank-four/BF16 case. The
CPU entry verifies original archive/tensor identities and an externally bounded
original epoch, then calculates both single-operand controls and the complete
input/upstream/interaction Gram decomposition. It never starts a model or device
operator, changes gradients or establishes training acceptance. This entry does
not transfer inputs; any staging is a separate exact-file operation.

## Development

Use Python 3.10 or newer. Before committing, run:

```sh
python tools/check_repository.py --worktree
python -m unittest discover -s tests -v
python tools/check_history.py
```

Run the component tests described in its README whenever changing its code. MLU
integration tests require the documented runtime and are separate from CPU tests.

Keep commits focused on one capability or fix. Include reusable regression tests
with changes in behavior. Generated results, performance reports, model weights
and exploratory scripts belong outside this repository. Commit messages use one
descriptive line. Preserve third-party copyright and license notices.

The repository checker enforces a source/configuration path allowlist, including
for untracked files in worktree mode. Root-level results, environment files,
reports, and unrecognized artifacts are rejected. Update the allowlist and its
tests when adding a legitimate source directory. Third-party source revisions
and license references are listed in `THIRD_PARTY_NOTICES.md`; no license for
redistribution of model weights or proprietary runtimes is implied.

To enable the repository check for staged files:

```sh
git config core.hooksPath .githooks
```
