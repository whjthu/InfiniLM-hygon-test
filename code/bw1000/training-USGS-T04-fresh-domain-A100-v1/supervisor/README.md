# Bounded T04 domain/control GPU supervisor — source only

This new source wraps the unchanged common native training entry. It does not add a training loop, alter the optimizer or RNG, or qualify Hygon. Root/GPU/CPU GO remain false until separate root adoption. No target execution was performed.

## Interface

Run the installed **actual Python file path**, not stdin, using the existing pinned A100 Python environment:

```
python bounded_GPU_supervisor.py --template ABS --template-sha256 HEX \
  --source-manifest-sha256 HEX --claim-sha256 HEX \
  --framework ABS --scope NEW_ABS --request-id HEX32 --publication HEX40 \
  --epoch ORIGINAL_CLIENT_EPOCH --a100-launch ABS --swift ABS \
  --models ABS --plugins ABS --master-port PORT
```

The root actual package owns Git publication verification, one-shot transport/install, source manifest and claim pins, exact runtime/template pins, resources and request namespace. This source is not that transport and must not be used to infer that an actual scope has been approved. The source gate requires future adopted root_GO/GPU_GO; the installed source directory is never an output directory.

Template is the domain observer config. Additional required fields are `source_publication`, `train_output=<scope>/train`, and `model_admission={path,bytes,SHA256}`. Its request/publication/epoch/output must equal the CLI; `deadline=epoch+820`, `maximum_seconds=820`. The original `model-files.json` inventory contains model_root/files/weight_stats. Its whole-weight SHA is prior evidence, reused only while bytes/mtime/inode/device exactly match. Metadata bytes are actually checked; this supervisor never rehashes full weight shards. Missing or changed evidence fails closed. Framework manifest must bind all merged source members (at least 400), including root's final entry, observer and cleanup verifier.

`supervisor` and `owned_cleanup` are generated in this new scope. The latter has `supervisor_source`, `held_child`, `owned_processes` pins, `scope_request_id`, and `anchor={PID,PGID,startticks,start_new_session,request_id,host,boot_id,uid}`. New-session common-entry bootstrap waits on stdin. Parent persists strong Popen anchor immediately, obtains actual identity, saves child-owner and config, then releases `GO` and config SHA. `execv` preserves PID/PGID/startticks. No fixed/fake rank is supplied. The common entry verifies actual direct parent and its own anchor before any admission/native launch.

## Budget, cleanup and evidence

All deadlines derive from the client epoch including upload/install: native820, supervisor stop850, held/tree cleanup880, post900. Normal completion returns early. TERM/INT enters finally. Strong held child cleanup precedes the unchanged owned-process registry cleanup. Its original fallback classifier cannot match the new plugin path; actual ownership here is seeded by the strong common-entry PID/startticks and its descendants. Only same-UID, exact recorded identities are signaled; no arbitrary GPU PID is killed.

The common entry performs the one fresh eight-card UUID/PCI/app0/port admission, with its own monotonic freshness bound after runtime probe. Supervisor post uses the same bound inventory and remaining original epoch. `release_permitted` requires known own-start state, reaped held root, no remaining owned tree, eight-card clean post and free port. Missing cleanup/post evidence cannot release. Timeout/transport UNKNOWN is never replay permission. The prelaunch fsynced `exec_request_attempted=false` changes to true before Popen; missing records are not evidence of no execution.

Fresh mode requires all 8 rank receipts with 100 microbatches/rank, 25 updates and step25. Resume requires 20 microbatches/rank, 5 updates, step25 and exact recovery true. Both bind mode/arm/pair_id/rank, retain all missing/failed ranks, and keep formal17/performance/effect false. The observer enforces initial504, actual input/cursor/token ledger, original optimizer/clip and checkpoint contracts; its completion is not a comparison of domain effectiveness. Per-rank receipt content is producer evidence; later independent consumer must check the actual artifacts.

Native stdout/stderr stay private, each capped at16MiB while monitoring. Rank/record reads cap2MiB. Private bounded exception message/trace is retained; public outcome contains only type, phase and hash/file pin. Supervisor imposes no inherited RLIMIT_AS or affinity change on native8-rank training; it requires actual inherited affinity2..64 cores excluding0,1. Root installer selects available cores excluding0,1, requires at least32 and caps64. Each poll sums only matched owned registry PID/startticks VmRSS using an opened /proc/PID directory and before/after identity checks. Unmatched identity does not read status. Sum256GiB is a sampled soft stop, never a hard RSS limit; shared pages may count more than once. Samples, peak, missing/changed identities and threshold state are persisted; threshold excess raises into original finally cleanup. No CPU peak or GPU performance claim is made.

## Source checks

Three qualified helper files are byte copies of `training-vlm-cuda-resume20-v1`; source proof pins them and old run_once. First new CPU boundary check stopped at the Linux-output fixture expressed with Windows Path (14 checks passed). That failure remains recorded. Only the fixture changed to PurePosixPath; the corrected new17 checks passed. No child was spawned, Torch imported, GPU queried, or older observer tests repeated. Actual clean-tree behavior and training remain future runtime evidence.

Root entry integration correction: the exact six receipt stages are planned, owned-cleanup-contract, fresh-admission, runtime, running, exited. Cleanup-contract actual must contain only held_session_identity_verified=true, cleanup_helpers_verified=true, resource_release_permitted=false (real booleans). Seven new branch-only checks passed; the earlier17 were not rerun.

Resource increment only: three new pure-stdlib fixtures (matched VmRSS/affinity, over-limit, identity mismatch without status read) passed. Earlier17/7 checks were not repeated. No actual process/RSS/GPU observation occurred.
