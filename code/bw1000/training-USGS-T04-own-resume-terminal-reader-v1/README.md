# Own-domain checkpoint20 to25 terminal metadata reader

Independent new source-only collector+offline audit. It reuses the qualified control collector's terminal-first/same-host boot PID/owned child logic and strict PCI canonical comparison. Root injects READ_PLAN.actual_CONTEXT and READ_DEADLINE<=now+60 and performs one ordinary A100 SSH read. Actual new request is bound, terminal/result SHA remains unfilled until genuinely observed. No launcher, Slurm, GPU query, Torch, state loading or new remote process poller.

Only `training-USGS-T04-own-resume20-A100-v1` is read. Collector first checks new saved outcome/config/child-owner/post/failure JSON, verifies request/epoch and same original hostname+boot/PID/startticks exit, then reads exact new8rank ledgers/receipts/restored-state-before-rng/RNG restoration point/tail-loss/resume comparison and one checkpoint25 small manifest plus actual new source manifests/RSS. A missing/live/cross-host original process remains UNKNOWN and stops before rank ledgers. No original domain remote reads; old rank100 ledgers are reused offline from local pinned603701B/7e34e226... bundle. No initial.pt, optimizer, adapter, scheduler or RNG tensor bodies are read.

Per rank required:20 actual new microbatches,5 real new updates21..25,save25 only,original20 restored adapter/optimizer/scheduler producer exact, original native _load_rng_state call-return exact, and each current input/forward/RNG/loss exactly matches that rank's old100 entries80..99. All8 rank final adapter/optimizer/scheduler and saved rank RNG producer comparison must be exact. Reader explicitly does not re-load typed state. Rank0 saved25==live is a frozen producer source condition before final receipt, not an independent body rehash. No fresh initial restoration or first2update delta requirement; no 800 new microbatches or25 new updates claim. On failure, progress and last finite native-return loss observation are retained even when final microbatches.json/receipt is missing; planned160 remains complete denominator.

Root saves private collector stdout directly to ignored local-only (cap8MiB,raw small files6MiB,each2MiB). Do not print it to the model: hashes/field metadata can include private IDs/inputs. Offline:

```text
<fixed-python> -B -X utf8 read_original.py --bundle <new-private-original> --bundle-sha256 <actualSHA> --context <actual-context.json> --old-ledger <existing-local-7e34-bundle> --output <new-private-audit.json>
```

Original domain supervisor FAIL/missingrank0 receipt/final optimizer live digest missing remain retained; a successful new recovery cannot retroactively make that run PASS. New5updates don't count as fresh25. Same-workload performance, heldout effect, formal17 and Hygon remain false. `check_tail_CPU.py` tests only one bounded tail20 and2corrupt-evidence cases, no fake complete model run or repeated old suites.
