# MLU VLM SFT and TextVQA

This package provides supervised fine-tuning entry points for Qwen3-VL, Gemma
3, InternVL 3.5, LLaVA-OneVision 1.5, and MiniCPM-V 4.6 on Cambricon MLU,
plus shared native TextVQA subset generation and official offline scoring.
It deliberately does not provide RL, PPO, GRPO, vLLM, or rollout entry points.


## TextVQA Validation Subset

All five registry keys (`qwen`, `gemma`, `internvl`, `llava`, `minicpm`)
share `scripts/eval_textvqa.sh`. It uses Swift 4.4.2's native
`TransformersEngine`, BF16, the model's native processor/template, and an
optional unchanged LoRA adapter. Generation is greedy, one question at a time,
with a 32-token budget and the fixed instruction "Answer the question using a
short phrase." Overlong prompts fail instead of being truncated.

This entry accepts only the pinned TextVQA 0.5.1 validation **64-question
subset**, not the full benchmark. `questions.json` has schema
`vlm-textvqa-questions-v1`, dataset/version/split/scope
`TextVQA/0.5.1/val/validation-subset`, and ordered `data` rows containing only
`question_id`, `question`, `image_id`, `image_path`, `image_sha256`,
`image_bytes`, `width`, and `height`. The manifest schema is
`vlm-textvqa-input-manifest-v1`; it binds the questions file SHA, ordered
question IDs, count, original annotations SHA, and subset SHA. Images are
relative to the explicit image root. Extra fields/files, answers, aliases,
missing images, wrong dimensions, duplicate IDs, and incomplete predictions
are rejected.

First run the CPU-only API preflight **inside the exact installed VLM image,
without device mounts**. It imports the pinned stack, constructs real Swift
request/config objects, and records installed implementation origins and SHAs;
it does not create a model. Its new output file must be outside the source:
`python3 -B runtime/textvqa_generate.py api-preflight --runtime-image "$IMAGE"
--out "$API_EVIDENCE"`. `IMAGE` must be a full `sha256:...` image ID, not a
tag. CPU fixture tests are not a substitute for this preflight.

Freeze identity against the actual read-only deployment files before admission.
Set the paths and file SHAs below to verified external artifacts. `TEMPLATE`
must match the installed native registry template in the API preflight;
an adapter snapshot must include its original `args.json`,
`adapter_config.json`, and `adapter_model.safetensors`.

```bash
export MODELS_DIR
common=(--questions "$QUESTIONS" --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA"
        --images-root "$IMAGES_ROOT" --api-evidence "$API_EVIDENCE" --runtime-image "$IMAGE"
        --adapter "$ADAPTER" --training-manifest "$TRAINING_PROVENANCE"
        --training-manifest-sha256 "$TRAINING_PROVENANCE_SHA")
MODELS_DIR="$MODELS_DIR" bash scripts/eval_textvqa.sh gemma freeze "${common[@]}" \
  --template "$TEMPLATE" --api-sha256 "$API_SHA" \
  --out "$IDENTITY_DIR"
```

`freeze` writes a new `identity.json`. It pins the complete model, adapter and
VLM source trees; installed API evidence; loader options; dataset manifest;
and generation policy. The native adapter's saved template/processor settings
remain in force. Optional `--processor-env '{"MAX_PIXELS":"262144"}'` freezes
that existing environment control; no unbound `MAX_PIXELS` is accepted.
Python and shell entries enforce the same registry model directory, model type,
native template, attention and plugin policy. Set `MODELS_DIR` to the verified
deployment root; individual model paths cannot override the registry.
For a base-model evaluation, omit the adapter and both training-manifest flags.

Adapters require a separate, externally verified JSON provenance file, not a
checkpoint directory label or an unbound training-data SHA. Its exact schema is
`vlm-textvqa-training-provenance-v1`, with status `accepted-qualified-training`;
fields are `model_alias`, `model_type`, `template`, `model_tree_sha256`,
`adapter_tree_sha256`, `checkpoint_args_sha256`, `adapter_config_sha256`,
`base_model_name_or_path`, `training_dataset` (the original saved-args list),
`training_dataset_sha256`, `training_source_tree_sha256`, `runtime_image`, and
`accepted_run`. The latter contains `run_id`, `host_label`, `global_step`,
`report_sha256`, `identity_sha256`, and the full `source_commit`. The trusted
provenance file SHA binds that accepted historical evidence; no acceptance is
inferred from filenames. Original PEFT config and args must agree with it and
the selected model/adapter bytes. Missing provenance fails before admission.
Identity v2 separately records its provenance-file and training-dataset SHAs.

The owned workflow currently supports **base-only evaluation on the main
`mlu-10000/node1` host**. Adapter generation remains diagnostic until its
training provenance and owned verifier are separately qualified. The auxiliary
host controller requires all eight idle cards; this single-card evaluator does
not bypass that policy.

The Docker entry must be a direct child of the reviewed host controller.
Use `runtime/textvqa_runner.py --runner-source "$RUNNER_SRC/genmlu/benchctl.py"
--out "$DEPLOYMENT"` to freeze its deployment inventory. All three controller
files must match the reviewed release policy in that module. A caller-supplied
hash cannot approve a different controller. Keep the deployment file outside
the source and record its actual SHA. Do not pre-create the run root.

Write the exact Python Docker launch arguments to an external JSON array
`$LAUNCH_ARGUMENTS`. Include `docker`, all required model/input/identity
arguments, `--runner-source`, `--runner-sha256`, `--runner-deployment`,
`--runner-deployment-sha256`, and `--out "$LEASE_ROOT/generation"`.
Resolve model alias/type, attention and plugin settings from the same registry
as the freeze command; overrides are rejected. Omit all adapter flags here.
The host helper validates this array and writes a new pre-launch plan:

```bash
python3 -B runtime/textvqa_host.py prepare \
  --launcher-arguments "$LAUNCH_ARGUMENTS" --launcher-arguments-sha256 "$ARGUMENTS_SHA" \
  --out "$PLAN_DIR/plan.json"
# PLAN_SHA is the printed plan SHA; save it before launching.
python3 -B runtime/textvqa_host.py run --plan "$PLAN_DIR/plan.json" --plan-sha256 "$PLAN_SHA"
python3 -B runtime/textvqa_host.py finalize \
  --plan "$PLAN_DIR/plan.json" --plan-sha256 "$PLAN_SHA" --output "$PLAN_DIR/result.json"
```

Only the exact questions and manifest files, independent images directory,
source, model, adapter, provenance file, frozen identity and API evidence are
mounted read-only. The training dataset and historical report roots are not mounted.
**Do not use the SFT Docker launcher or mount the ground-truth subset parent.**
The Docker path uses runc, no network, UID/GID 1001:1002 by default, private
2 GiB shared memory, an 80 GiB host-memory limit, four CPUs, and bounded threads.
Triton uses an executable ephemeral cache in `/tmp`; source and model mounts
remain read-only. It requires existing
device/control nodes without changing permissions. Before model construction
it sets and reads back the MLU allocator budget; both the readback and reserved
peak must stay at or below 75 billion bytes. Each prediction binds its question
and image and records actual native-forward tensor devices and finite outputs;
model parameters must be on logical `mlu:0`. Failures cannot become completed
generation evidence.

The lease and numbered device nodes retain the host's physical card ID.
With only that card mounted, both visibility variables use container-local
ordinal `0`, not the physical ID. This applies only to the isolated TextVQA
entry; the multi-card SFT launcher keeps its existing device-list behavior.

Score in a **separate CPU process**, with the original annotations and answer
subset mounted only there. Pin `generation.json`'s actual file SHA.
The supplied SFT JSONL must contain nonempty `images` lists; all referenced
image bytes and optional question IDs are checked for subset overlap.

```bash
python3 -B runtime/textvqa_score.py \
  --questions "$QUESTIONS" --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
  --images-root "$IMAGES_ROOT" --identity "$IDENTITY_DIR/identity.json" --identity-sha256 "$IDENTITY_SHA" \
  --api-evidence "$API_EVIDENCE" --run-root "$LEASE_ROOT/generation" --run-sha256 "$GENERATION_SHA" \
  --annotations "$OFFICIAL_ANNOTATIONS" --subset "$ANSWER_SUBSET" \
  --training-data "$TRAINING_DATA" --training-data-sha256 "$TRAINING_DATA_SHA" \
  --training-manifest "$TRAINING_PROVENANCE" --training-manifest-sha256 "$TRAINING_PROVENANCE_SHA" \
  --training-images-root "$TRAINING_IMAGES_ROOT" --out "$SCORE_DIR"
```

The unchanged official MMF `TextVQAAccuracyEvaluator` performs answer
normalization and ten-annotator leave-one-out soft accuracy. Its fixed upstream
commit, source SHA, and BSD license are in `runtime/TEXTVQA-NOTICE.md`.
The output names its 64-question denominator and byte-level held-out scope
relative to the supplied SFT inputs; pretraining overlap is not assessed.
Generation and score v2 records use `diagnostic_completed` and always set
`owned_execution_accepted=false`. The execution receipt is context, not a
credential: a direct Python caller cannot grant itself lease acceptance.
The host helper requires Python 3.10+, Pillow, the Docker CLI and the
reviewed controller's host dependencies. It sets the isolated controller
import root, uses two idle checks 300 seconds apart, watches every 15 seconds,
and allows one hour for the child. A failed controller run is not finalized.

The packaged host verifier binds the reviewed deployment, pre-launch plan,
actual completed lease and exit status, registered container ID and unique
lease nonce, source/input inventories, result SHAs, released lease and container
absence. The result accepts owned execution only, not quality or performance.
This is a trusted execution workflow, not protection against arbitrary code
running under the same host UID. A score does not assert full-benchmark quality,
recovery, or speed.

Scoring defaults to the raw answer. For the pinned MiniCPM non-thinking
template, `--answer-policy empty-think-prefix-v1` explicitly removes exactly
one leading empty `<think>\n\n</think>\n\n` envelope. Other models, nonempty
reasoning, malformed envelopes and nested tags are rejected. Original
predictions are never changed; `answers.jsonl` records raw/extracted answers,
and the score records the extraction policy and file/source SHAs. Keep raw
and extracted scores separate. Neither protocol changes the MMF evaluator.

## Build

Build from a base image that already contains the site-approved Torch MLU
runtime. Pin the complete image digest at deployment time; the Dockerfile does
not install a driver or `torch_mlu` itself.

```bash
docker build \
  --build-arg BASE_IMAGE=registry.example/mlu-runtime@sha256:<digest> \
  -t vlm-sft:swift4.4.2 .
```

`requirements-mlu.txt` pins the verified Swift/Transformers stack. Installation
uses `--no-deps`, so it cannot replace the base image's Torch MLU runtime. The
build verifies the installed distribution metadata, `swift.__version__`, and
`torch==2.6.0+cpu` with `torch_mlu` importability. The base image must provide
Python 3.10 and a driver-compatible Torch MLU runtime.

The build context admits only the runtime, plugins, scripts, tests, and listed
metadata. Local outputs, weights, reports, and environment files are excluded.
To test the actual Docker COPY rules without installing dependencies or using
an accelerator, run `bash tests/test_image_context.sh`. It builds a temporary
scratch image and inspects its exported filesystem without starting a process.

## Train

Models default to `/mnt/models`; override that location only with
`MODELS_DIR`. An output directory is required and must be outside this source
tree. The default visible-device list is eight cards, `0,1,2,3,4,5,6,7`.

```bash
DATASET=/mnt/datasets/example.jsonl \
OUTPUT_DIR=/mnt/outputs/qwen3-vl-sft-001 \
bash scripts/train_sft.sh qwen
```

To use a selected subset, pass its exact device list unchanged:

```bash
MLU_VISIBLE_DEVICES=0,1,2,3 \
DATASET=/mnt/datasets/example.jsonl \
OUTPUT_DIR=/mnt/outputs/gemma-sft-001 \
bash scripts/train_sft.sh gemma
```

Run inside the container with the same environment forwarding:

```bash
DATASET=/mnt/datasets/example.jsonl \
OUTPUT_DIR=/mnt/outputs/llava-sft-001 \
bash scripts/docker_run.sh bash scripts/train_sft.sh llava
```

`docker_run.sh` mounts the source and `MODELS_DIR` read-only, mounts
`DATA_MOUNT_ROOT` read-only when a dataset is supplied, and mounts only the
new external output run directory read-write. It resolves paths on the host,
rejects output overlap with read-only inputs after resolving aliases, creates the run directory exclusively, and
does not mount all of `/mnt` or an output parent read-write.

## MiniCPM Formal Identity

Ordinary MiniCPM SFT accepts unrecorded model identity. For a formal comparison,
first freeze a new manifest outside the source tree, then require it for the
comparison run:

```bash
python3 runtime/verify_minicpm_identity.py \
  --model-path /mnt/models/MiniCPM-V-4.6 \
  --write-manifest /mnt/evidence/minicpm-wrapper-weights.json
MINICPM_STRICT_IDENTITY=true \
MINICPM_IDENTITY_MANIFEST=/mnt/evidence/minicpm-wrapper-weights.json \
DATASET=/mnt/datasets/example.jsonl \
OUTPUT_DIR=/mnt/outputs/minicpm-formal-001 \
bash scripts/train_sft.sh minicpm
```

The documented `docker_run.sh` entry point accepts the same two variables. In
strict mode it requires an absolute regular manifest outside the source and
model trees, mounts that exact file read-only, and forwards only those controls.

The command creates the manifest exclusively, so an existing comparison record
cannot be replaced. It pins only the formal wrapper files and the complete
MiniCPM model tree, including `config.json`. It does not identify installed
`ms-swift`, Transformers, or MiniCPM modeling implementation code; formal
framework/runtime identity must be recorded separately.

### Qualified MiniCPM 25-step preset

MiniCPM-V 4.6 has one explicit opt-in preset named `qualified-gc-25`. It fixes
the qualified eight-rank BF16 LoRA workload: batch 1 with accumulation 4,
length 4096, 262144 image pixels, 25 optimizer steps with 5 warmup steps,
cosine AdamW, LoRA rank 16/alpha 32/dropout 0.05, and checkpoints at steps 10,
20, and 25. The preset does not include diagnostic callbacks or make a
performance, recovery, or acceptance claim.

Strict MiniCPM wrapper/model identity remains an explicit optional runtime
control and is required for qualification use of this preset. A newly generated
manifest is required after wrapper changes. The preset rejects resume, extra
Swift arguments, and conflicting ordinary workload environment values; a fixed
field may be unset or exactly match the preset. Its only variable workload
parameter is `MINICPM_PRESET_GRADIENT_CHECKPOINTING`, which must be `true` or
`false` and defaults to `false` only while this preset is selected. Ordinary
`train_sft.sh` behavior is unchanged when `MINICPM_SFT_PRESET` is unset.

```bash
MINICPM_SFT_PRESET=qualified-gc-25 \
MINICPM_PRESET_GRADIENT_CHECKPOINTING=false \
MINICPM_STRICT_IDENTITY=true \
MINICPM_IDENTITY_MANIFEST=/mnt/evidence/minicpm-wrapper-weights.json \
DATASET=/mnt/datasets/example.jsonl \
OUTPUT_DIR=/mnt/outputs/minicpm-qualified-gc-off \
bash scripts/docker_run.sh bash scripts/train_sft.sh minicpm
```

The pinned image/parser, source and data identities, actual 186-module LoRA
scope, MLU initialization, and native checkpoint directories at steps 10, 20,
and 25 remain external qualification gates. In particular, this wrapper does
not infer that native Swift writes `checkpoint-25` merely because `max_steps`
is 25, and it does not import the diagnostic callback used by qualification.

The Docker launcher runs as the invoking host UID:GID and forwards its
supplementary numeric groups, so checkpoint files remain host-readable and
host MLU device-group access is retained. It requires a POSIX `id` command.
`HOME`, Hugging Face, and Transformers caches are writable ephemeral paths
under container `/tmp`; keep `OUTPUT_DIR` writable by the invoking host user.
The launcher bypasses the vendor image entrypoint because it writes SSH files
under `/root` even for jobs that do not use SSH. With no command, it retains the
image's interactive `bash` default.

The wrapper maps each validated `MLU_VISIBLE_DEVICES` item to
`/dev/cambricon_dev<N>` plus `/dev/cambricon_ctl` and `/dev/cambricon_ipcm`.
Set `MLU_DOCKER_DEVICE_PREFIX`, `MLU_DOCKER_CONTROL_DEVICES`,
`MLU_DOCKER_RUNTIME`, or comma-separated `MLU_RUNTIME_MOUNTS` for a site with
a different Cambricon device/runtime layout. The wrapper fails before Docker
starts if a required node or mount is absent.

## Resume

Resume is intentionally available only through `RESUME_FROM_CHECKPOINT`; it
is not an extra command-line option. Set it to an absolute, existing
`checkpoint-N` directory containing `trainer_state.json` and
`adapter_model.safetensors`, `optimizer.pt`, and `scheduler.pt`, and use a
distinct `OUTPUT_DIR`. A one-process checkpoint needs `rng_state.pth`; when
`NPROC_PER_NODE=8`, the DDP checkpoint needs exactly `rng_state_0.pth` through
`rng_state_7.pth`. Weights-only checkpoints are rejected. `docker_run.sh`
creates the output directory exclusively; `train_sft.sh` also accepts its
already-created empty directory.

```bash
DATASET=/mnt/datasets/example.jsonl \
RESUME_FROM_CHECKPOINT=/mnt/outputs/qwen3-vl-sft-001/v0-20260919-120000/checkpoint-100 \
OUTPUT_DIR=/mnt/outputs/qwen3-vl-sft-002 \
bash scripts/docker_run.sh bash scripts/train_sft.sh qwen
```

The pinned image verifies that its actual `swift.SftArguments` Hugging Face
parser exposes `--resume_from_checkpoint`. The launcher passes that flag only from the
validated environment value. Docker resolves and mounts only the exact
checkpoint directory read-only; it neither remounts its parent nor permits
the new output directory to overlap the checkpoint (including aliases).

The LLaVA plugin is loaded only for `llava`. It registers a model-specific
checkpoint key mapping and initializes two OneVision rotary buffers after the
native model loader has materialized the model. For the pinned Transformers
runtime, it additively registers the missing `default` RoPE factory only when
that factory is absent; it does not replace an existing factory. It contains no
trainer hooks, diagnostics, telemetry, or result writing.
LLaVA-OneVision defaults gradient checkpointing to `false` unless
`GRADIENT_CHECKPOINTING` is explicitly set.
Gemma-3-4B defaults to `false` only with devices `0,1,2,3,4,5,6,7`, eight
processes, BF16 LoRA, frozen vision encoder, length 512,
`MAX_PIXELS=262144`, per-device batch 1, and accumulation 4. Other shapes
default to `true` until qualified. `GRADIENT_CHECKPOINTING` explicitly overrides
either default. The qualified recipe also uses rank 16, alpha 32, dropout
0.05, `all-linear` targets, packing and padding-free disabled, and right padding.

Extra SFT tuning arguments use a strict allowlist to preserve the output
contract. Supported options are learning rate/schedule/weight decay, steps and
epochs, batch/accumulation/checkpointing, vision and sequence limits, logging
and save cadence, LoRA settings, validation dataset, seed, evaluation/save
strategy, and run name. Use the exact underscore long-option spelling, with a
separate value or `=` value. Configuration-file arguments, option
abbreviations, positional extras, and environment argv injection such as
`RAY_SWIFT_ARGS` are rejected.

Frozen benchmark settings that are not part of the default training command
use explicit environment bindings. The launcher accepts the following names:

```text
SPLIT_DATASET_RATIO  DATA_SEED  LR_SCHEDULER_TYPE  OPTIM
ADAM_BETA1  ADAM_BETA2  ADAM_EPSILON  WEIGHT_DECAY  SEED
DATASET_SHUFFLE  DATALOADER_DROP_LAST  PACKING  PADDING_FREE
PADDING_SIDE  TEMPLATE  AVERAGE_TOKENS_ACROSS_DEVICES  ADD_VERSION
```

Each name maps to the same lower-case Swift option, for example
`ADAM_BETA1` to `--adam_beta1`. Boolean values must be exactly `true` or
`false`, and `PADDING_SIDE` must be `left` or `right`. Unset bindings do not
change the default command. The launcher rejects invalid values and rejects a
setting supplied through both its environment binding and extra arguments.

Set `MLU_DOCKER_NAME` when an external lease manager tracks the container by
name. The launcher validates it before passing a single `--name` argument.
Set `MLU_DOCKER_RUN_ID` to the lease manager's run identifier so Docker receives
the ownership label `genmlu.run_id`. Arbitrary Docker labels are not accepted.
For a benchctl-managed single-container run, set the name to the declared
`--container-name` and set the run ID to the exact `--run-id`, for example:

```sh
MLU_DOCKER_NAME=genmlu-run-1 MLU_DOCKER_RUN_ID=run-1 scripts/docker_run.sh ...
```

Both values accept ASCII letters, digits, `_`, `.`, and `-`, must start with a
letter or digit, and are passed without normalization. The run ID may contain a
single character. This interface does not implement the additional lease label
required when one benchctl run owns multiple containers.

## CPU Contract Test

The test uses fake `swift` and Docker executables and no accelerator. It
validates all five registry entries, missing-model rejection, output-overwrite
protection, controlled resume validation and read-only checkpoint mounting,
container environment forwarding, and importability of the LLaVA plugin.

```bash
bash tests/test_cpu_contract.sh
```

The LLaVA registration boundary also has a CPU-only pytest suite. Run it in
the verified image, where the pinned Transformers and Torch MLU packages are
available:

```bash
python -m pytest -p no:cacheprovider tests/test_llava_loader.py
```

### Controlled native LoRA dropout diagnostics

`vlm.runtime.lora_dropout_session.LoRADropoutSession` records actual native
masks by rank, update, micro-forward, module name and call ordinal. Each packet
binds the complete structured model input (including labels and image bytes),
model/recipe identity, input shape/stride/dtype and a SHA256-protected reference
manifest. Capture leaves native output unchanged. Replay still consumes the
backend's native RNG draw before substituting the real recorded mask. It never
sets dropout to zero or infers a mask from nonzero outputs.

The `native_lora_dropout_alignment` external Swift callback initially supports
only the frozen MiniCPM recipe with gradient checkpointing disabled, eight
ranks, four accumulation micros and the first two optimizer updates. It requires
all 186 native LoRA dropout modules once per micro-forward, then restores their
original methods and removes its hooks. It checks the original initial trainable
parameter identity and still requires the native 25-update run to finish. This
callback does not qualify checkpointed recomputation on other VLM recipes.

Select the existing launcher options `--diagnostic-plugin` and
`--diagnostic-callback native_lora_dropout_alignment`; supply the configuration
path through `NATIVE_LORA_DROPOUT_CONFIG` and its exact bytes SHA256 through
`NATIVE_LORA_DROPOUT_CONFIG_SHA256`. The configuration fields are `case`, `mode`,
`recipe_SHA256`, `initial_trainable_SHA256`, `world_size`, `forwards`,
`accumulation`, `expected_dropout_modules`, `mask_byte_budget`, and `output`.
Replay additionally requires `reference` and
`reference_manifest_SHA256_by_rank`. Output/reference roots are private run
storage, with separate `rank-00` through `rank-07` directories. Obtain reference
manifest pins from a completed, independently checked capture before publishing
the replay claim; do not modify an executing capture or overwrite its packets.

Mask copies, hashing, packet I/O and diagnostic callback time are extra costs.
These runs must be labeled diagnostic and excluded from formal throughput.
The real PEFT/AdamW CPU regression test is
`python -m unittest discover -s vlm/tests -p test_lora_dropout_session.py -v`.
It covers two optimizer updates, full pre/post-clip gradients, optimizer state,
RNG progression, malformed identities and failure denominators. Small CPU test
models are engineering fixtures, not MiniCPM numerical or quality acceptance.


For controlled MiniCPM replay, optional `initial_state` contains the original
`path`, `bytes` and `SHA256` of `initial.pt`. The callback validates fresh step
zero and an empty optimizer owning exactly the selected parameters, checks the
complete source inventory before copying, then verifies the destination named
parameter digest. It loads only original trainable tensors, leaving frozen base
parameters and destination CPU/device RNG untouched. This is initialization,
not checkpoint/optimizer/RNG recovery. The original MiniCPM file pin is required.

Repeat `--diagnostic-plugin` and `--diagnostic-callback` to supply matching
ordered observer pairs through the unified entry. Put
`native_lora_dropout_alignment` before the original first-update recorder
(`original_first_updates`, class `FirstUpdates`) when loading initial state.
The callback rejects missing, duplicate or late recording order. Multiple
observers are recorded as an ordered `diagnostic_observer.observers` list;
single-observer receipts keep their existing shape. These are diagnostic runs,
with all added initialization/copy/hash/observer costs excluded from formal
performance. CPU checks: `python -m unittest discover -s vlm/tests -p test_initial_trainables.py -v`.

When restoring the original trainable initialization, pass
`--diagnostic-callback-position before` with the ordered callbacks
`native_lora_dropout_alignment`, then `original_first_updates`.
Both callbacks then run before the original baseline callback, so its initial
parameter digest observes the restored values. The initializer rejects a
baseline callback that already ran before it. The default `after` placement
preserves existing observer-only launch behavior. Diagnostic runs retain all
capture overhead and do not qualify formal performance.
