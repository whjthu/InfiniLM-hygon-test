This is the first connected ordinary GLM public-provider candidate. It is not a
benchmark driver: there is no fixture plan, auto-client, ABBA, request-count
limit, alternate math, second scheduler, H6/speculation, or model-type alias.
It serves complete admitted requests through the real latest99 public routes.
No Model/GPU qualification has been run for this new composition.

Actual call chain:

```text
python -m public_glm_native (stage1)
  -> real official AutoInfinilmProcessor + actual r424 prepare_engine
  -> original prepared sampling/history + profile middleware
  -> real public InferenceServer._create_app(engine_factory=GLMProvider)
  -> public lifespan -> GLMProvider.start()
     -> actual spawn Process actors + same-process emitter owner Thread
     -> owner_main -> original Native/source/object/MD chain verification
        -> visible GPU0 Common import, original Torch/hardware/owner setup
        -> actual static CAPI bind + original topology/create
        -> original registry WORLD vote BEFORE weight I/O
        -> original partition/keys/trim/NoV census
        -> TP8/WORLD16 exact warmup + all16 >8GiB BEFORE weight load
        -> actual r421 complete raw checkpoint load + Native finish
        -> all16 loaded/>4GiB, exact providers/helpers and strict affinity
        -> r423 actual loop policy agreement/barrier
        -> all16 immutable owner-ready records -> public lifespan returns
  -> public HTTP request -> same original bridge/NativeRequest object
     -> actual owner loop/control/prefill/decode/reset/token completion
  -> public lifespan stop -> close admission + stop intent + active cancel
     -> original group-boundary vote and STOP drain
     -> original pending cancellation AFTER global drain
     -> reset, actual destroy, cache/host/M15/registry checks, shutdown
     -> join known Native actors only; HTTP is never joined by Native
```

Stage0 invokes the same CLI with stage0, generates the original communicator
UIDs, spawns8 real owner processes, and supervises them. Stage1 spawns7 owner
processes and one owner thread in the public process. Only that thread receives
the same prepared queue object. Spawn arguments contain rank plus JSON config;
NativeRequest locks, processor, queues, tensors and Native pointers are never
pickled into MPQueue. All helpers live in one `public_glm_native` package with
static relative imports. No BASE slicing, source exec or installed-module
monkeypatch is used by production code.

`config.template.json` is deliberately `execute_authorized=false`. A necessary
future execution claim must bind the exact source manifest/config, original
epoch, actual Slurm step, resource membership and fresh device inventory before
using `--serve`. This source task grants no new GPU execution. Example shape:

```sh
PYTHONPATH=/path/to/this/package python -m public_glm_native \
  --config /path/to/bound-config.json --config-sha EXACT_CONFIG_SHA256 \
  --source-sha EXACT_MANIFEST_SHA256 \
  --stage 1 --run UNIQUE_RUN --actual-step JOB.STEP \
  --gpu-start-epoch ORIGINAL_EPOCH --serve
```

The counterpart stage0 must be launched concurrently on its allocated peer
node, under one bounded fail-stop Slurm step (original kill-on-bad-exit=1), with
the original64CPU/8DCU visibility/environment. Existing exact C1/TP8/PP2 layout
and two qualified node names remain enforced; nothing silently widens this
scope. Qualified DTK/library paths, BLAS preload, shca network, width4 owner
binding and original env requirements remain mandatory. This is not a portable
new Native build. It reuses exact Native952, SDK and compiled dependencies.

Source-only validation omits `--serve`; it loads no installed module or DSO and
does not mean constructor readiness. Ordinary routes retain their default
AsyncLLMEngine when `engine_factory=None`. The isolated public copy adds trusted
provider injection and instance stream/request timeouts (defaults100/1000);
this provider passes the original prepared3600/7200, preserving long-prefill
timeouts without globally replacing a server module.

Original service and Native deadlines use the same caller-supplied GPU epoch:
4500s service/weight-I/O admission, 5100s Native cleanup hard bound. No start,
retry, wait or stop slides that epoch. Shutdown closes admission, sets an
emitter stop event and marks cancel intent without finishing requests. The
original loop owns group votes, reset and finish. FIFO sentinel alone is not
used as the stopping strategy. Joins are bounded; failed actors terminate only
known owned Process objects. A stuck same-process Native CCL thread ultimately
uses original whole-leader fail-stop rather than unsafe thread cancellation.
Normal joins use at most the original600s cleanup reserve and never exceed the
original Native deadline; a known failed group gets at most20s for local unwind.

Qualification boundaries:

- New CLI/public lifecycle/actor composition is source and local-recording CPU
  evidence only; actual installed imports/Native cold create/load have not run.
- The local test consumes actual Actors, Run, GLMProvider, real prepared queue,
  actual changed public constructor/lifespan and real postdestroy counter gate.
  Spawn execution/Native owner body/FastAPI app construction are recording
  leaves. Its16 ready records are test outputs, not16 actual device owners.
- Existing r421/r422/r423/r424 tests are reused in their prior scope. They are
  not repeated or promoted to a new whole-model result.
- Native952/NoV16 retains only its existing850 two-tool and414 scoped Model
  result. New GSM, broad benchmark, public HTTP/SSE, online15tps and this new
  emitter-thread composition remain unqualified.
- Only GLM is wired here. Deep qualified-default selection and Kimi public
  provider integration remain separate unfinished work; experimental HC1 or
  TP8PP4 performance is not silently selected for either family.

Actual publication/Model execution and external node PID/KFD/port accounting
remain for the later precise resource claim/controller. This package does not
submit Slurm, acquire allocations, kill unknown PIDs, auto-reload weights or
rewrite running source. Device/cold-load failures must be preserved and stopped.
