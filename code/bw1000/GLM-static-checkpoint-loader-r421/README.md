Static GLM checkpoint-loader extraction; no model/provider readiness claim.

`checkpoint_loader.py` contains the actual BASE CAPI declarations and complete
raw checkpoint load algorithm as explicit Python functions. It has no import-
time Torch/device/env/CDLL/model side effects and does not parse or execute a
benchmark source file. `bind_library(path, SHA256)` is an explicit DSO boundary;
`declare(lib)` is the same binding contract with a supplied actual library.

`load_checkpoint(...)` consumes the actual Native `keys(ctx)`, checks that every
key exists in the checkpoint index, groups by file, sorts files/names, reads
contiguous CPU tensors, views FP8 E4M3 bytes as UInt8, submits original dtype
codes F32=0/BF16=1/U8=2, flushes at 2048 and each file end, then calls finish.
No FP8 dequantization, recompression, dtype conversion, parameter remapping,
tensor transpose, head replacement, or partial-key shortcut is introduced.
The original caller's model path (including its validated weight-cache overlay)
is explicit. Existing phase markers, three deadline call boundaries and report
fields remain. The real three-tensor dtype metadata sample is not a complete
owner key set or a weight payload.

`r414-static-loader-integration.patch` shows the exact proposed consumer delta
against r414 owner `model_entry_r290.py`; it has **not** been applied to r414,
r417, or any installed source. It removes both BASE string executions and the
loader AST guard, uses the static functions, preserves owner Torch thread
settings and explicit stage/tag, and leaves construction, metadata/key checks,
head/shard policy, admission, native math, reset and context cleanup untouched.
The real owner `OwnedContext` scope still owns failure cleanup. This module
must not independently destroy a caller's context on a failed load.

BASE prefix inventory:

- imports including Torch/safetensors/Transformers, now dependencies of the
  real owner instead of implicit `exec` globals;
- Torch thread/TF32/reduced-BF16 settings, retained by the proposed owner;
- absolute root/model and Slurm stage/tag variables, now explicit owner inputs;
- CDLL open and CAPI declarations, now explicit binding functions;
- `check()` and communicator-ID `unique()`, preserved on `Binding`;
- old diagnostic worker and `__main__`, never imported or executed here.

Local CPU consumer uses real checkpoint index/header records (BF16 lm_head,
FP8 down projection, F32 scale), real Python/ctypes host addresses, and the
actual new load function. Torch storage, safetensors payload, SDK file/maps and
Native CAPI leaves are recording substitutes. It covers success and refusal/
failure/deadline propagation; it does not load weights or validate device math.

Remaining constructor work is explicit: actual configuration and immutable
provider identity; owner/device/communicator construction; all-key/head/shape
checks; complete load/finish/cache and all-rank agreement; queue admission;
forward/cancel/reset; normal/error shutdown. Public HTTP must own its lifespan.
The existing r417 owner starts and joins HTTP itself; putting it unchanged
behind a public provider that joins that owner can deadlock by mutual joins.
The eventual Native actor must own only queue/world/reset/destroy and be
stoppable/joinable by the public server. That lifecycle separation is not
implemented or claimed by this extraction.

Native952/NoV16 qualification remains its original two-tool/scoped-pair range;
this source-only extraction grants no additional quality or model readiness.
