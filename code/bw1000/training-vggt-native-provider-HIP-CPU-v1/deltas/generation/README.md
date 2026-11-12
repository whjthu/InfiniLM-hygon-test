# GenMLU generation runtime

MLU inference backends and bounded training and RL workflows for seven generative and geometry models.

## Installation

Use a vendor PyTorch and `torch_mlu` image compatible with the selected model sources. Install the package and model-specific source dependencies in that image:

```bash
python -m pip install -e '.[runtime,test]'
scripts/apply_source_patches.sh /path/to/official-sources
```

The pinned upstream revisions and origins are in `sources.lock.json`. Entries marked `reference-only` document algorithm provenance and are not required runtime checkouts. The source checkouts and model weights remain outside this repository. Pass source names after the root to patch only selected checkouts, for example `scripts/apply_source_patches.sh /sources hy-motion`. The patch script supports linked Git worktrees and rejects mismatched upstream revisions. The CLI checks that the source directory exists but does not enforce the entire source lock at startup; deploy from checkouts verified by the patch script and record their identities separately. The default patch set covers the packaged backends, not the upstream Hunyuan command-line trainers or TRELLIS BiRefNet pipeline.

To verify all shipped patches against the locked upstream revisions without
changing a source cache, run `bash scripts/verify_source_patches.sh /sources`.
It uses temporary clones and applies each source's patches twice to check
idempotence. Set `TMPDIR` to a scratch filesystem with sufficient space.

### Release image

Build a wheel from a clean checkout, then stage an external image context from
that same commit. The wheel carries its source identity; the staging command
rejects dirty, unknown, or mismatched builds. Run from the repository root:

```bash
python -m pip wheel ./generation --no-deps --no-build-isolation \
  --wheel-dir /release/wheels
source_commit=$(python generation/scripts/prepare_image_context.py \
  --wheel /release/wheels/genmlu-0.1.0-py3-none-any.whl \
  --output-dir /release/image-context)
docker build /release/image-context \
  --build-arg BASE_IMAGE=registry.example/mlu-runtime@sha256:<digest> \
  --build-arg SOURCE_COMMIT="$source_commit" -t genmlu:release
```

Use the actual wheel filename and a new context directory. The base image must
already contain compatible runtime dependencies; the image installs the wheel
without resolving dependencies or replacing vendor Torch. Deployment scripts,
configs, and patches are copied from the frozen commit, not from the workspace.
Weights, datasets, results, and upstream source checkouts are mounted separately.

## Included capabilities

The included registry has independent entries for SDXL, Z-Image Turbo, HunyuanVideo 1.5, HY-Motion 1.0, TRELLIS 2, VGGT, and Stable Fast 3D. The command dispatcher loads only the backend selected by the model and command. TRELLIS exposes latent generation, bounded CPU decoded fields, and a decoded-cache textured-GLB export component with separately supplied native dependencies. Complete end-to-end output requires preservation of all native sparse-structure coordinates. RL is limited to SDXL, Z-Image Turbo, HunyuanVideo 1.5, and HY-Motion 1.0.

Multi-process verl orchestration and HunyuanVideo 6+1+1 orchestration are not provided by this package.

| Component | Supported boundary |
| --- | --- |
| VGGT | Geometry inference; camera-head training with a frozen aggregator, or bounded camera-backbone training of the aggregator plus camera head. |
| Stable Fast 3D | Material-head training with `foreground_material_targets`; this is not a full supervised official training workflow. |
| TRELLIS 2 4B | Staged 512 latent inference; bounded CPU decode and decoded-cache opaque textured-GLB export. Complete output is conditional on no native-coordinate subsampling. No full-model training or RL entry point. |
| TRELLIS shape-flow 1.3B | Separate single-card component-training entry point with 12 explicit LoRA targets and an identity-checked latent cache; not full 4B training. |
| SDXL DDPO | Uses the engineering `image_color_reward`; it is not a learned production reward. |

For multi-image Stable Fast 3D inference, set `pipeline_cpu_sidecar: true` and
`cpu_sidecar_workers: 2` to prepare mesh UVs in two bounded CPU workers while
the owner handles MLU inference, texture baking, and ordered export. The default
remains one worker; the two-worker mode targets throughput, not single-image
latency. CPU stages are measured separately from MLU work.

These are bounded component-training workflows, not general-purpose production trainers. Checkpoint saves remain enabled. Post-training repeat-update recovery experiments are disabled by default; set the YAML boolean `validate_recovery: true` for SDXL DDPO, Z-Image training, VGGT training, SF3D training, HY-Motion training, or Hunyuan training/RL to run them explicitly. Disabled checks and their metrics are `null`, not passed. These experiments validate same-process reload behavior; SF3D material-head training separately supports the fresh-process interface below.

### Recovery interfaces

| Workflow | External recovery entry point |
| --- | --- |
| SDXL LoRA training | `resume_from_checkpoint`: complete native `checkpoint-N` directory produced with `full_checkpoint: true`; automatically enables the full contract and rejects legacy checkpoints without its completion manifest. |
| SDXL DDPO | `resume_from_checkpoint`: full-state checkpoint produced by this backend. |
| HY-Motion MotionRFT | `resume_from_checkpoint`: full-state checkpoint produced by this backend. |
| Z-Image Flow-GRPO | `resume_from_checkpoint`: full-state checkpoint produced by this backend. |
| VGGT camera training | `resume_from_checkpoint`: placement-preserving full-state checkpoint bound to the selected `training_scope`; camera-head and camera-backbone checkpoints are not interchangeable. |
| HunyuanVideo native training | Default single-rank v1 resumes from `checkpoint-N/state.pt`. Explicit experimental DP2 uses its committed v2 directory; fixed DP8 uses a separate DP8 schema. Single-rank, DP2, and DP8 checkpoints are not interchangeable. All bind model, cache, configuration, source, and global optimizer step. |
| HY-Motion full training | `resume_from_checkpoint`: full-state `checkpoint-N/state.pt` produced by this backend and validated against its full-training contract. Native HY-Motion training requires exactly one card. |
| Z-Image Turbo training | `resume_from_checkpoint`: full-state `checkpoint-N/state.pt` from this backend, accepted only for a single MLU process with identical model tree, preset adapter, CSV/image dataset, training configuration, source, runtime, and scheduler identity. It restores the completed global optimizer step, next dataset cursor, trainable state, buffers, AdamW state, scheduler state, and Python/NumPy/CPU/MLU RNG streams. |
| SF3D material-head training | `resume_from_checkpoint`: full-state material-head checkpoint from this backend with matching source, weights, inputs, configuration, optimizer, and RNG identity. It restores the exact global optimizer step, next data cursor, optimizer state, and frozen-feature cache. |
| TRELLIS shape-flow component | `resume_from_checkpoint`: complete optimizer-step `.pt` checkpoint, restricted tensor/primitive loading, with identical plan, source, weights, cache, package, and runtime identities. No in-process replay is performed. |
| Hunyuan RL | No external recovery entry point in this version; a non-null `resume_from_checkpoint` request is rejected before output creation or runtime setup. |

Keep the original optimizer schedule, source dependencies, weights, and workload
when resuming, and use a new run name and output directory. Checkpoint formats are
backend-specific; an adapter alone is not sufficient for equivalent continuation.
For SDXL LoRA, independently compare optimizer, scheduler, RNG, and data-position
recovery across fresh processes. Acceptance of a checkpoint does
not replace a fresh-process next-update equivalence test.

Z-Image training accepts an optional `checkpoint_cpu_threads` limit (integer
1 through 256; default `null`). It only limits synchronous full-checkpoint
construction, then restores the training thread count before serialization or
the next update. Use it only on the main thread without concurrent Torch CPU
work. It does not change training batch size, precision, checkpoint frequency,
or the process-wide training policy. The configuration and helper source are
part of checkpoint identity; keep them unchanged when resuming. This option
does not apply to native Swift checkpoints or Z-Image RL.

### SDXL native LoRA device evidence

Native SDXL LoRA training records the actual UNet input, output, and
trainable-parameter devices for every optimizer update. The wrapper requires
ordered global steps and one consistent MLU device, rejecting missing evidence
or CPU/main-forward device mismatches. This proof covers LoRA training, not
full-UNet training. Recovery support is limited to the full-checkpoint contract
below; it does not establish performance or quality for other workloads.

`full_checkpoint: false` is the default and leaves fresh training unchanged.
Opt-in full checkpoints require one MLU process, `dataloader_num_workers: 0`,
`train_batch_size: 1`, `gradient_accumulation_steps: 1`, and a pinned runtime image.
External resume automatically requires this contract: legacy native checkpoints
and adapter-only exports are rejected, not silently treated as full-state recovery.
The restricted loader restores validated state before the next update and records
an exact pre-update witness; independent fresh-process comparison is still required.

### VGGT camera training scopes

`training_scope: camera_head` remains the default and preserves the existing
frozen-aggregator workflow. `training_scope: camera_backbone` trains the
upstream aggregator and camera head together while depth, point, and track
heads remain disabled. This is bounded camera-backbone fine-tuning, not full
all-head VGGT training.

Camera-backbone training requires exactly one card and rejects
`cache_frozen_aggregator_features: true`, because cached aggregator outputs
would bypass backbone gradients. It preserves upstream parameter flags, so
parameters such as the unused DINO patch-embedding mask token remain frozen.
The preset is `configs/vggt-camera-backbone-training.yaml`.

The qualified MLU runtime uses a CPU fallback for the DINO positional-embedding
antialiased bicubic backward operator. Model forward remains on MLU; this scope
does not imply that every backward operator runs on the accelerator.

Each checkpoint contains the complete selected trainable range under an
explicit `camera_head` or `camera_backbone` state key; the final state artifact
uses the matching filename.
Results report pre-clip camera-head and backbone gradient norms and independent
camera-head and aggregator SHA values. Resume identity includes the scope,
cache mode, optimizer parameter names, initial weights, data, and source
identity.

`training_scope: camera_depth_point_backbone` trains the aggregator plus the
camera, depth, and point heads with the official VGGT multitask loss. It retains
the CO3D loader's depth, camera-space points, world-space points, and point mask,
then applies the official CPU geometry normalization before transfer to MLU.
Missing or invalid targets are rejected. The track head remains disabled because
this workload has no track targets and the public VGGT track loss is unsupported.
This is supervised geometry fine-tuning, not full all-head training.

The preset is
`configs/vggt-camera-depth-point-backbone-training.yaml`. Frozen-aggregator
feature caching is forbidden. Checkpoints and forward-device evidence cover the
aggregator and all three enabled heads. The DINO antialiased bicubic backward
CPU fallback remains visible, so successful training does not mean every
backward operator ran on MLU.

External performance comparisons must use identical validation settings and timing boundaries. Z-Image RL pins the model and differential-adapter identities in its example; verify those inputs and update the identifiers when changing weights.

### HunyuanVideo Native DP2

`configs/hunyuanvideo-data-parallel-training.yaml` enables only fixed two-rank,
plain-Tensor data parallelism with SP1. Supply two distinct `--cards` and the
actual immutable `training_cache_sha`. The parent owns the lease and launches
two workers with torchrun, no restart, and a bounded process timeout. Each rank
selects its logical device and initializes CNCL before constructing Muon.
Pass `--source-root /sources` with the native checkout at
`/sources/hunyuanvideo-1.5`. DP2 rejects a symlink or junction source root or
checkout, and any symlink or junction inside the native source tree.
Unlike the single-rank workflow, DP2 requires `validate_recovery: false`;
in-process recovery replay is unsupported.
The upstream cache, LoRA preparation, noise/timestep sampling, and flow-MSE
are shared with the single-rank backend. Gradients are averaged across both
ranks before clipping and the unchanged native Muon step. Missing gradients
in the trainable range reject the run, rather than silently excluding parameters.

Native Muon is replicated, not sharded, for plain Tensors. Both rank files
contain the complete trainable state, buffers, modes, optimizer moments and
branch flags, consumed-gradient counts, and rank RNG. Only rank zero publishes
`checkpoint-N/COMMITTED.json`, after both atomic rank files validate and their
placement-normalized model/optimizer bytes agree. Resume requires that directory,
not either individual rank file. All ranks preflight before application; RNG is
restored last, with no extra warmup updates. Partial transactions are rejected.
FSDP, DTensor, sequence parallelism above one, elastic restart, and world-size
changes are unsupported. The existing single-rank launch guard remains in force.

This implementation does not constitute hardware or recovery qualification.
Before delivery, run fresh two-rank native training and independent CP10/20
continuations, checking both ranks' next update and full suffix state/RNG.
CPU tests use the actual pinned Muon source only when
`HUNYUANVIDEO_MUON_SOURCE` points to its external file with the expected SHA256;
otherwise those tests skip. Allocator caps and recorded peaks are per process
and do not assert a driver-wide physical-memory bound or performance acceptance.

### HunyuanVideo Native DP8

`configs/hunyuanvideo-data-parallel-8-training.yaml` explicitly selects fixed
eight-rank plain-Tensor DP with SP1. Supply eight distinct `--cards`, the exact
`training_cache_sha`, and the same source-root layout described above. Both
visibility variables must match the ordered cards; each worker uses its logical
rank device. Gradients use SUM divided by eight before clipping and native Muon;
missing trainable gradients on any rank reject the run. Every rank stores full
state and its own RNG with the fixed single-cache cursor. Rank zero publishes
only after all eight receipts and replica states agree. Restore requires all-rank
preflight, applies state first, and restores RNG last without extra warmup updates.
DP2 remains the default and retains its v2 schema; DP8 uses
`hunyuanvideo-dp8-replicated-full-step-v1`. Cross-world/schema or source-identity
changes reject recovery. This path does not add FSDP, SP above one, RL, or elastic
restart, and does not establish hardware, recovery, or performance qualification.

### SDXL rollout graph

SDXL DDPO includes optional MLU graph rollout support and keeps `rollout_graph: false` by default. Enabling it requires the MLU graph SDK and captures only the fixed-shape raw UNet rollout in BF16; the Actor remains eager, in-place LoRA updates are visible to replay, and pre-existing forward and dispatch/offload hooks are rejected. Checkpoints bind the selected mode and graph SDK identity, so mismatches are rejected. Capture initialization is reported separately from the measurement window.

```yaml
rollout_graph: true
```

### Optional validation

Z-Image training and RL use `validate_turbo_regression: false` by default.
Enable it to generate and compare before/after eight-step images; the provided
training and RL examples enable it explicitly. Disabled metrics and checks are
`null`. Keep this setting unchanged in performance comparisons. Changing the
regression contract invalidates an existing RL resume identity.

HY-Motion RL gradient traces are opt-in. `validation_trace_max_steps` defaults
to 20 and `validation_trace_max_bytes` to 67108864 bytes of stored tensor
payload. Once either limit is reached, further diagnostic records are dropped
without changing training; the trace records the dropped count. This is not a
limit on total Python process memory.

`configs/hy-motion-rl-full-gc-off.yaml` is an explicit checkpointing-off preset
for the listed Full-model, four-sample, four-sampling-step MotionRFT workload.
It requires the pinned AFS inputs and upstream source identities in the config.
The general `hy-motion-rl.yaml` retains checkpointing by default. Do not apply
the preset's memory or speed expectations to other sequence lengths, batches,
rewards, or model variants without a separate comparison.

SDXL LoRA accepts positive `checkpointing_steps` and `checkpoints_total_limit`
values (defaults 10 and 3). Fresh runs save the initial adapter separately from
rotating checkpoints, so weight-change validation does not depend on retaining
checkpoint-10. Resumed runs also export the loaded adapter before the first
update. Comparisons hash tensor names, shapes, dtypes, and values, excluding
SafeTensors file metadata; the input checkpoint's file SHA is recorded separately.

Inference repeat determinism is unmeasured (`null`) in benchmark mode or with
fewer than two comparable repetitions. A valid single SF3D result does not
require enabling repeat validation. Explicitly failed checks still fail the run.

## Running

Every run requires an external output root and a unique run name. Existing run directories are rejected before a backend is started.

```bash
genmlu infer \
  --model sdxl-1.0 \
  --config configs/sdxl-inference.yaml \
  --model-root /models \
  --output-root /outputs \
  --run-name example-001 \
  --cards 0 \
  --lock-root /shared/genmlu-locks \
  --lock-host physical-host-01
```

Pass `--weights /path/to/model` to override the model-root layout. Commands that execute upstream code also require `--source-root /path/to/official-sources`. `genmlu eval` fails closed because a complete evaluation workflow is not provided. The CLI intentionally has no runtime-smoke mode.

Run the CLI only in an environment whose MLU device exposure has already been configured by the deployment platform. Every non-dry run requires `--cards`, an absolute host-shared `--lock-root`, and an explicit physical-host `--lock-host`. `--cards` acquires the host-scoped per-card lease and initializes `MLU_VISIBLE_DEVICES` when it is unset; an existing different value is rejected. For containers, mount the same host lock directory and supply the same physical-host identifier to every container on that host.

## Component boundaries

The package owns dispatch, run isolation, manifests, checkpoint helpers, and the MLU adaptations in `patches/`. Model architectures, data preparation, reward weights, and upstream source dependencies remain external. Supported RL commands call the committed algorithm backends only; unsupported model/command combinations terminate with an error instead of substituting a synthetic result.

Every backend returns artifacts, logs, metrics, stage timings, and checks, including single-sample inference. Failed correctness checks cause a nonzero exit and a failed manifest. Optional telemetry is recorded under `observations`; unavailable utilization does not invalidate an otherwise successful command. A completed command is not a claim that unmeasured quality, performance, or recovery checks passed.

TRELLIS requires an explicit `output_contract`: `latent_only` writes staged latent evidence as its primary NPZ artifact, while `diagnostic_multibox` additionally writes a non-production multibox preview. Set `decode_cpu_fields: true` with `latent_only` or `full_mesh` to launch an isolated CPU subprocess after latent persistence. The subprocess strictly loads the pinned clean TRELLIS.2 shape and material decoders, emits seven shape fields, six PBR attributes, coordinates, and dual-grid vertices/faces in an additional NPZ, and enforces a configured point bound without truncating the supplied cache. The UV/PBR exporter is independently meaningful for a verified decoded cache, but `full_mesh` may be declared complete only when the generation backend proves that the decoder received the entire native sparse-structure coordinate set. Selecting a count-limited subset invalidates complete image-to-3D output even if component export validation passes. The generation backend currently samples only the 512 flow branch, so this entry point requires `cpu_decode_resolution: 512`; it rejects 256 and 1024 before runtime setup. The checkpoint's default resolution and actual geometry resolution are recorded separately. The standalone CPU decoder can also process explicitly supplied 256/1024 latents, but the generation entry point does not provide those sampling branches. In `latent_only`, all formal/full/textured-mesh flags remain false. Full TRELLIS 4B training and TRELLIS RL remain unsupported; the separate bounded shape-flow component training entry point is described below.

### CPU Textured GLB

`full_mesh` requires an explicit `max_sparse_tokens` admission budget (8 through
32,768); it does not inherit the latent/diagnostic default of 1024. Choose this
capacity for the intended workload before launch. It is not a truncation target
for `full_mesh`. If the native 32-grid support exceeds it,
the backend saves the rejected coordinates and stops before shape or texture
flow loading. It never raises the budget or switches output contracts.
Bounded `latent_only` and diagnostic outputs retain their existing selection
behavior but record native, retained and dropped counts. The support artifact
contains the ordered native coordinates, selection indices and each flow's
input/output coordinates. The manifest checks these arrays against the actual
latent artifact; a count or a reported boolean alone is not sufficient.

`output_contract: full_mesh` additionally requires `decode_cpu_fields: true`,
resolution 512, an explicit `cpu_texture_export.dependencies` identity, and
evidence that native sparse-structure coordinates were not subsampled.
See the commented generic configuration in `configs/trellis2-inference.yaml`.
There is no bundled native dependency, private filesystem default, model download,
or implicit diagnostic fallback. This contract describes an opaque PBR artifact,
not unrestricted upstream pipeline parity, model quality, speed or training support.

Supply a CPU build of the SF3D Unwrapper and TextureBaker interfaces referenced at
`ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2`. The dependency root contains exactly the
two Python files and one native `_C` extension for each package. Declare the full
six-file SHA256 inventory and exact Python/Torch/NumPy/Pillow versions. Actual
file identity, import origin and small native
unwrap/rasterize/interpolate calls are checked in an isolated subprocess before
MLU initialization. A module-spec lookup alone never qualifies a dependency.

The exporter preserves all decoded triangle positions and ordering, expanding
every triangle to independent corner indices before native UV unwrapping. Each
corner uses its face's winding-derived geometric unit normal, not an average
across adjacent faces. This isolates nonmanifold connectivity without moving,
adding or removing triangles; it changes index connectivity and uses flat
shading. Unreferenced input vertices do not enter UV alignment or normal
calculation. The report records original vertex/face counts, unreferenced
vertices, boundary edges, edges incident to more than two faces and maximum
edge incidence. These are input diagnostics, not a watertightness, manifoldness
or mesh-quality claim. Degenerate triangles still fail before native UV calls.
`uv_layout: native` is the default and uses the pinned SF3D unwrapper.
`uv_layout: face_tiles` explicitly selects the bounded `face-tiles-v1`
implementation instead. It assigns one right-triangle tile to every input face
with grid `isqrt(face_count - 1) + 1`, centered integer pitch, and fixed
`padding_pixels: 4`; an atlas too small for the required inset fails before UV
allocation and is never enlarged. Every triangle edge is a seam, every face
receives equal atlas area rather than uniform world-space texel density, and the
common right-triangle parameterization can distort source triangles. This is a
deterministic exporter layout, not an official unwrap. Both layouts require
interior texel coverage for every face before baking and after GLB reparse.
The report records the selected layout, implementation version, and, for
`face_tiles`, its derived grid, pitch, margin, inset, and side.
It samples the six-channel sparse material field at
original-world surface positions, then applies the declared `[x,z,-y]` export
rotation. Sparse FP32 sampling uses the eight-corner, half-voxel-center forward
rule of FlexGEMM `6dd94a859c26ee8246888502eada3dd8ad85532e`, including bounded-grid
lookup and valid-neighbor renormalization. FlexGEMM is not imported or required;
this is reference-compatible mathematics, not measured CUDA bitwise parity.

GLB embeds lossless RGBA base color and RGB metallic-roughness `[0,R,M]` PNGs.
Base color uses glTF sRGB interpretation and material data uses linear
interpretation, with no added gamma conversion. Alpha bytes are retained but
the material is explicitly OPAQUE. No normal map, AO, transparent rendering,
hole filling, decimation, remeshing or BVH projection is provided. Invalid
geometry/UVs, overlapping atlas interiors, unsupported material neighborhoods,
dependency mismatches and exceeded budgets fail closed.

`texture_size` defaults to 1024 (power of two, maximum 4096); `padding_pixels`
defaults to 4 and `island_padding` to 0.02. `chunk_size` defaults to 65536,
`max_points`/`max_faces` to one million each, `max_raster_work` to 50 million
triangle-bounding-box texel tests, `max_bytes` to 2 GiB and `timeout_seconds` to
1800. Arrays are bounded before allocation and observed child peak RSS must
also fit `max_bytes`. No dense 512-cubed material fallback exists. The atlas
uses native point samples; tiny triangles can contain no texel at the selected
resolution, without being deleted from geometry. Texture filtering is linear
without mipmaps; uncovered atlas padding never replaces covered material data.

On Linux, peak RSS is the isolated export executable's `/proc/self/status`
`VmHWM`, including its imports, native preflight and export allocations since
exec. It is not the parent inference process or a combined process-tree peak.
Linux `getrusage().ru_maxrss` can retain the pre-exec address-space peak and is
not a fallback. Missing, malformed, nonpositive or wrong-unit `VmHWM` fails
closed. The default budget remains 2 GiB. macOS alone retains the byte-valued
`resource` process-lifetime fallback; other platforms report unavailable.

The primary artifact becomes `trellis2-textured.glb` only after component
validation and the parent no-native-coordinate-subsampling gate.
Latents, decoded fields, decode report, texture config, compact pixel/triangle/
barycentric correspondence NPZ and export report remain manifest-bound artifacts.
The report is atomically published last and SHA-bound, not cryptographically
authenticated. Parent and manifest validation re-read the actual GLB, check its
restricted single-mesh glTF 2.0 structure, compare transformed geometry, verify
UV correspondence/interior coverage, and recompute both entire PNG pixel arrays
from the original material field. Source, dependencies, settings and all artifact
hashes must agree. Partial outputs cannot acquire successful full-mesh flags.

### Shape-flow component training

Use `genmlu train --model trellis-shape-flow-1.3b` with a completed external copy
of `configs/trellis-shape-training.yaml`, `--weights /models/trellis-2-4b`, and
the usual output, lease, card and source-root arguments. The YAML source path
must equal `<CLI source-root>/trellis2`. The placeholder configuration fails
closed until cache and source identities are supplied. No experiment cache
path or measured cache hash is built into the runtime.

The component uses the official 512-branch shape-flow checkpoint, latent grid
32, and all 640 base keys, preserving native BF16/FP32 module dtypes. Only the
12 named attention projections in `trellis_training_contract.TARGETS` receive
rank-2, alpha-4 LoRA: 24 FP32 tensors and 96,256 elements. This is not all-layer
fine-tuning, full TRELLIS 4B training, decoder training, or RL. Actual attention
must execute SDPA on MLU in BF16; every trainable gradient is checked before
clipping and the update. Initially zero A gradients are permitted, but the
combined gradient norm must be nonzero and finite.

The single NPZ must contain int32 `coords[N,4]`, float32 `shape_latent[N,32]`,
and float32 `condition[1,L,1024]`. Coordinates must be unique, batch zero, and
inside the 32 grid. File order is preserved without sorting or truncation.
`max_tokens` and `max_condition_tokens` are required rejection budgets, each
bounded by 32,768. The optional `texture_latent` and `structure_latent` arrays
are validated but not trained. The declared representation is denormalized
shape SLAT; the pinned pipeline mean/std normalize it in FP32 before BF16
transfer. `cache_origin_kind` is an explicit `generated` or `real` declaration,
not a verified quality claim. A generated example is not real-data pretraining
or an Objaverse evaluation.

Cache semantic SHA256 covers the three training arrays in sorted name order:
UTF-8 name, ASCII `str(shape)`, ASCII `str(dtype)`, then C-contiguous bytes.
Source inventory SHA256 is `json_sha` of the mapping from every actual
`trellis2/**/*.py` relative POSIX path to its file SHA256, including untracked
Python files. Supply that hash and the exact Git revision after applying the
required SDPA source patches. Both are verified before import; actual file,
tree and patch identities are recorded. Cache file, ordered semantics,
normalized semantics, dataset declaration and normalization identities are
bound into each checkpoint along with full weight and executed-package identity.

`max_steps` includes warmup updates; defaults are 25 total and five warmup.
Warmup only excludes those initial global updates from timing; LR is constant.
`checkpoint_steps` selects unique positive global completed steps within the
total plan (default milestones 10 and 20 when within that plan). This save
frequency is part of the workload and resume identity. A checkpoint is also
saved at the requested stop/final boundary. An external next-update comparison
can use `[10, 11, 20, 21]` in every process to compare full state at 11 and 21
without any extra update. Set `stop_after_global_step: 10` or `20`
without changing the total plan to prepare an independent continuation. In a
fresh process and new run directory, set `resume_from_checkpoint` and its
required `resume_checkpoint_sha256`, and remove or advance the stop boundary.
The actual input checkpoint path and SHA are recorded in the result. Container
names and execution labels are excluded from the training contract; actual
runtime versions and device identity are bound separately. Resume
restores trainable tensors, buffers, module modes, full AdamW moments and their
placements, Python/NumPy/CPU/MLU RNG, and the completed single-cache data cursor.
The checkpoint must validate before any state is applied. Older adapter-only
checkpoints are rejected; no unsafe pickle fallback or hidden repeat update
exists. Native MLU RNG validation must support an independent generator clone
or resume fails closed. A restored boundary does not itself certify independent
next-update equivalence, performance, or model quality.

When the MLU checkout contains required runtime patches, set `cpu_decoder_source_root` to a separate clean checkout at revision `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. If omitted, the sidecar uses the normal TRELLIS source root and still rejects any dirty or mismatched checkout. The manifest records the normal configured source identities separately from `provenance.cpu_decoder_source_identity`. Optional `cpu_shape_decoder_sha256` and `cpu_material_decoder_sha256` values pin the full decoder safetensors; actual full-file hashes are recorded even when those expected values are absent.

Manifests record `provenance.executed_source` from the package contents separately from configuration-supplied source labels. Builds from a Git checkout also record its commit, tree, and dirty state in the wheel. Archive builds without Git metadata report unknown commit/tree rather than trusting a supplied label. Installed package contents are checked against the build digest; the digest covers Python sources and packaged YAML, not external upstream checkouts or weights. `config_sha256` identifies the resolved run configuration, including run-specific fields, and is not a cross-run workload identifier.


### Native VGGT first-update evidence

The optional `VGGT_FIRST_UPDATE_BINDING` plus its SHA256 enables the shared
CUDA/HIP recorder on the existing camera-head training path. The binding checks
the complete original config, actual trainable manifest, frozen aggregator,
pretrained weight/input/source identity and one real accelerator. It requires
the original online aggregator,224pixel two-frame inputs,five warmup plus20
measured updates,and global checkpoints10/20/25. Default execution performs no
capture I/O; diagnostic execution retains all original native math and timers.

The first update writes actual initial parameters,realized input/RNG,derived
pose target/mask,all four pose stages/loss,pre/postclip gradients,updated
parameters and first AdamW state. Parameters and optimizer tensors are stored
one trainable at a time,at most256MiB per artifact and12GiB in total,so the full
camera head does not require one multi-gigabyte host snapshot. Files and failed
partial plans remain private on the experiment host. Subsequent native updates
continue to25; the completion marker requires the original frozen aggregator
hash to remain unchanged. Step failures retain the planned denominator and
completed file hashes; other native/checkpoint failures also require the job's
independent terminal/accounting audit and cannot be inferred from a first-step
capture alone.

All capture/hash/synchronization cost remains in enclosing diagnostic wall
windows. These records qualify neither formal performance nor fresh-process
recovery,held-out quality or cross-platform numerical agreement. The current
CPU tests use tiny real Linear/AdamW tensors; full-model structure,actual device
execution and independent readers are separate qualifications.
