This is an isolated patch against the deployed latest-family public Python
source (upstream identity `99f5c90a40f9b2546bd0202cba3397ed1d55c5a0`). It does
not modify the installed SDK, the C++ registry, or the running r417 package.

The first callable boundary is `InferenceServer(..., engine_factory=callable)`.
The factory receives exactly the existing engine settings inside the real
public application lifespan. `None` selects the original `AsyncLLMEngine`.
There is no fallback from a failed explicit provider into a different model.
This is trusted application configuration, not a request field or an arbitrary
CLI import string. No CLI Native selector is claimed in this change.

For a resident Native owner whose real construction, key validation, complete
weight load, finish, cache allocation and all-rank agreement already succeeded:

```python
server = InferenceServer(
    model_path=real_weight_path,
    max_cache_len=32768,
    max_batch_size=1,
    temperature=0.0,
    engine_factory=lambda **settings: resident_native_bridge,
)
app = server._create_app()
```

This uses the original public routes/lifespan instead of reconstructing a
second FastAPI application in `existing_server_app`. A queue's existence or
`NativeEngineBridge.start()` is **not** proof of model readiness. The example
requires the real resident owner, retains its owner-thread create/load/forward/
reset/destroy contract, and does not itself implement that owner. Factory
partial-construction failures must clean up before raising. Engine `stop()`
must signal and join/drain the owned service; existing bridge `stop()` alone
only enqueues a sentinel. Therefore this patch is an implemented public seam,
not a completed three-family public Native backend.

Required remaining production consumers, using existing qualified operations:

| Boundary | GLM | Deep | Kimi |
| --- | --- | --- | --- |
| construction/metadata | real per-rank GLM model CAPI and actual keys/partition; 16 or 32 rank agreement | actual `modern_native_engine_create`; keys/layout from real loader | typed ordinary create, capabilities, keys/layout/cache_geometry, exact head policy |
| complete weight load | extract existing complete loader from r414 `model_entry_r290.py`; it currently slices/execs absolute baseline source | extract r366 `bootstrap_r206b.py` actual BASE consumer; no new selector math | r410 parameter_loader + complete key coverage and typed load/finish |
| forward | r417 owner reset, bounded complete prefill, actual forward and all-rank control; preserve W4/NoV profile identity | r366 existing forward and `native_decode_r206b.generate` with cancellation/finite checks | r410 ordinary kind0 prompt<=512/kind1 decode1; no H6/spec qualification |
| cache/reset | real model-owned C1 32768 state, original reset and WORLD drain | real Native compressed/indexer/rolling state reset; no Python fake KV | real original engine cache/ledger; ordinary-only geometry and reset |
| shutdown | all owners reset/drain/destroy, then join and only then stop frontend | same owner thread reset/destroy; preserve normal/error distinction | all 8/node workers destroyed, full PP control shutdown |

These are different owner topologies and output contracts. A generic
`ModelRunner` adapter cannot be obtained by aliasing model types or returning
the queue from `InferEngine`. The minimal next implementation should factor
the already qualified owner construction/load/finish portion into a normal
provider constructor with explicit topology/config and immutable provider
identity. It should expose readiness only after actual all-owner success,
then supply the existing bridge through this public seam. Migrating the GLM
AST-sliced loader is still real work; r417 is a transition entry, not that
completed constructor.

Baseline selection remains separate from this public seam. Deep's current
experimental `976f`/HC1 owner has a recorded 5/6 tool-quality failure; its source
is useful for lifecycle extraction but is not a qualified default. The
`afc6`/HC0 independent semantic68 run1221 is the separately qualified historical
scope. Kimi's `6eb`/TP8PP4 two-request ordinary validation is not the same as
the qualified `8df`/TP16PP2 agent/performance scope. GLM's `952`/NoV16 proof is
the scoped850/414 work; real new GSM generation remains pending. None of these
qualifications transfers to the generic public constructor by adding a hook.

The current private emitter owner starts and joins its own HTTP thread. It
cannot simply be wrapped by a public lifespan whose `stop()` also joins that
same Native owner: those joins could wait on each other. A real provider must
leave HTTP ownership with the public server and own only the Native service
actors, their queue/control, drain/reset/destroy and join. This remains work;
the present patch does not implement a partial provider or claim readiness.

Run the local boundary check with fixed Python, the exact server source and
existing bridge source as arguments to `check_CPU.py`. Its result explicitly
lists recording leaves. It does not import installed FastAPI/Torch, dlopen a
Native library, construct/load a model, or qualify HTTP/SSE. No new job ran.
