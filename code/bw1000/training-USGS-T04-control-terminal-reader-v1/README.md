# New T04 generic-control original terminal reader

Source-only standard-library collector plus offline saved-JSON audit. No actual request, publication, epoch, terminal SHA or success is invented. Root supplies the actual CONTEXT `{request_id,publication,source_manifest_SHA256,epoch}` and READ_DEADLINE <= now+60 before one read-only execution of `collect_original.py` through the existing A100 ordinary SSH. Root owns all dispatch; no SSH launch code or new GPU is part of this reader.

The collector first reads only new `training-USGS-T04-fresh-control-A100-v1/run/{outcome.json,domain-config.json,child-owner.json,after.json,failure-private.json}`. It validates the original request/epoch and original supervisor+child hostname/boot/PID/startticks. Across host or missing outcome/owned identity/live original process => UNKNOWN, no ledger read. Absence never means not executed. Only after actual terminal+known original processes absent does it read each exact rank's small JSON, source manifests and four bound small references. This is ordinary A100 process accounting, not Slurm terminal accounting; there is no new Slurm call or GPU query.

`READ_PLAN.json` lists exact names and budgets. Individual files <=2MiB, total consumed raw small files <=6MiB, stdout private bundle may be <=8MiB. It contains private ledgers and original user inputs; redirect it directly into ignored local-only, never print to the model/public/Git. No initial.pt, adapter/optimizer/scheduler/rank-RNG body or original domain large state is read. Original initial.pt is only stat-ed under its one exact path; all source/reference SHA are checked against actual new published source and actual runtime config, which the root's original publication admits. The old domain pairref551B is a pinned dependency only, not replaying original training/audit.

After root saves the actual original bundle and SHA:

```text
<fixed-python> -B -X utf8 read_original.py --bundle <private-bundle> --bundle-sha256 <actualSHA> --context <actualcontext.json> --output <new-private-result.json>
```

Full result contains bounded original small-file SHA rows and no raw inputs/tokenIDs. Root may publish the bounded summary of actual state/8rank/800 planned and observed/25 updates/finite inputs/strict native45/control800/45unique/EOS/masks/tokens/global occurrences/restoration records/checkpoint pins/ownedpost/RSS. FAIL, missing ranks and missing files remain in the denominator. Output never upgrades generic-control to same-workload formal performance, Hygon, formal17, recovery or heldout effect.

The frozen producer has two evidence limitations: (1) original native callback checks initial-load RNG before/after unchanged but does not save separate raw initial RNG snapshots; (2) rank0 frozen `finish()` requires typed saved25==live adapter/optimizer/scheduler before writing final receipt but saves no separate comparison numeric JSON. Reader records these as producer-source+receipt conditions, not independent new RNG/typed-state rechecks. Original failed domain/rank0 missing remains failed; no old original run or tensor audit is repeated. Initial restoration must be an actual nonempty recorded dict, named504/43646976 must match original domain, seed alone is insufficient.

The only new fixture checks UNKNOWN/missing planned800 and wrong-context/body-read rejection. It does not synthesize a successful model run or repeat RSS fixtures.
