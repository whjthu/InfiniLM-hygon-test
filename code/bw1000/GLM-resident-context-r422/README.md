`ResidentContext` is a real per-rank contextmanager, not a public model factory.
It is isolated from r414/r417/installed code. No model or DSO is loaded by
importing it. No HTTP, process spawn, scheduler or cross-thread join is present.

The caller provides an already strictly bound Native952 library, actual model
path/config identity, the validated original C1/TP8/PP2/32768 NoV layout,
stage/rank/communicator IDs, original absolute deadline callback, phase callback,
original process identity, real Torch/safetensors and explicit record paths.
The entering thread must already own the selected actual device. The original
topology validator still enforces its qualified host/layout/env scope; this
change does not silently generalize it to different nodes or topology.

Inside the entering thread, the contextmanager:

1. checks explicit and actual-process layout/config identity, original helper
   process identity, and original exact Native952 provider proof before create;
   never changes the caller's environment;
2. calls the original `topology_r404.create_context`, then immediately enters
   the original `OwnedContext` with its already-created pointer;
3. checks the original Native partition/layer range, performs the original
   positive unloaded trim, verifies keys were unchanged, and calls original
   `constructor_snapshot` for exact Native952 provider and NoV cache census;
4. calls the complete static r421 checkpoint loader and original Native finish,
   rechecks the Native key set and exact loaded-parameter count;
5. yields the **active** handle, never a pointer returned after leaving `with`;
6. on body/entry failure or normal exit, calls the actual reset CAPI and then
   the original exactly-once destroy/shutdown helper. Primary errors survive
   reset/cleanup errors; normal cleanup failure is visible. Wrong-thread exit
   cannot consume cleanup and the original owner can still close its scope.

Usage boundary:

```python
with ResidentContext(binding, model_path=weight_path, layout=layout,
                     rank=rank, stage=stage, comm_ids=ids,
                     environ=actual_env, hostname=actual_hostname,
                     torch=torch, safe_open=safe_open,
                     deadline_check=original_deadline_check,
                     phase_marker=phase_marker,
                     lifecycle_path=unique_lifecycle_path,
                     constructor_record_path=unique_constructor_record_path,
                     report=report) as resident:
    # Actual Native forward belongs here, on this owner thread, with the
    # original typed buffers/state contract. Do not retain resident.ctx.
    use_actual_owner_context(resident.ctx)
```

The example's forward consumer is deliberately not implemented by this card.
It is not a stub invoked by the module. Constructor/load/reset/destroy are
implemented and actually consumed by local CPU checks, with Native leaves
recorded. The per-rank primitive must be called concurrently with correctly
coordinated peers: the underlying real create uses real communicators.

Head/weight semantics are unchanged: the strictly bound Native constructor's
complete key set is authoritative, including embedding/norm/lm_head keys on
the original stage. All keys must be in the real index and all are sent through
the original loader; Native finish checks completeness and performs original
weight processing. No guessed head alias, replacement head, FP8 conversion or
new shape math is added. Non-layer Native keys are reported for audit.

Four reused modules are byte-identical: original topology/create, OwnedContext,
NoV admission/provider census, and static r421 loader. Their hashes and original
paths are in AUDIT.json. This is ordinary static module reuse, not worker/BASE
AST extraction. Record paths must have existing parents and be unique.

Local CPU checks consume actual contextmanager and all these actual helper
functions. Native/provider/Torch/storage leaves are recorded. Their 39 key names
are selected from the real index; tensor shapes derive from actual config for
recording only, not re-read tensor payloads or actual Native key enumeration.
The real config SHA is pinned. Lifecycle, constructor and checkpoint-index
files are real temporary files, including error and normal cleanup records.

Still unimplemented and **not** qualified: all-rank registry agreement, collective
preload warmup/admission, original per-owner memory floors, complete head/shard
and provider inventory across peers, first/final real affinity, post-destroy
bank/cache reclamation observations, global ModelReady, forward/queue control,
public CLI/server startup and process-level postcleanup. Local finish does not
replace any of those original gates or mean Model_READY. They remain required
in the eventual resident-provider orchestrator. HTTP must own its own lifespan;
do not place the old HTTP-owning r417 worker behind a reciprocal join.

Native952/NoV16 only retains its existing r414/850 two-tool/scoped validation.
No new benchmark/quality/Model/GPU evidence is created by these CPU checks.
