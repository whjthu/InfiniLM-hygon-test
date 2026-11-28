# LLaVA CLEVR Val16 single-DCU owner

Use the exact hosted parent `1103606`, node `f13r1n14`, and normal login-side Python `/usr/local/python3.12/bin/python3`. This scope never yields, signals or cancels an incumbent. Any numbered controller or queue step on the candidate node vetoes dispatch. The only signals are to the confirmed process identities that this run launches.

## Source and media stage

```sh
/usr/local/python3.12/bin/python3 -I -S -B /approved/host.py stage \
  --request "$request" --host-sha "$host_sha" \
  --native-source /approved/native --support /approved/previous-llava-owner \
  --reload-tool /approved/llava_clevr_cp25_reload.py
```

The native source directory needs the frozen manifest plus only the worker and contract. The support directory needs the exact original `owner.py`, `snapshot.py`, `slurm_accounting.py`, and `owned_processes.py` from `training-llava-clevr-first2-owner-v1`. The installer exclusively writes `R/incoming/llava-clevr-5000-val16-<request-prefix>-stage`. It verifies the real staged `requests.jsonl`, `IMAGES.json`, and 16 PNGs, then hardlinks just those 18 files. The extra stage readback receipt and any other files remain outside the guest input mount. The installer emits `stage_SHA256`. Stage consumes zero Slurm steps.

## Fresh grant and single invocation

Use a new 32-hex request. Prepare a root-reviewed release receipt with exactly `request`, `parent`, `node`, `holder` as the latest actual numbered nine-column UTC accounting row, `independent_physical_clean` and `worker_returned` both true, and `source_SHA256` of the actual independent release evidence. Holder accounting must be exact `COMPLETED 0:0` for n14 and one of the admitted resource profiles. Supply the original proof only when the real release evidence exists.

Create the fresh root grant with exactly these keys

- `request`, `parent`, `node`, `stage_SHA256`, `prior_holder` as the same original row, and `prior_release_SHA256` of the release receipt
- `epoch`, `deadline` in HOST UTC seconds with more than 180 and at most 2250 seconds of total budget
- `root_GO`, `ordinary_excluded` and `idle_release_clean` all true on current evidence
- `boot`, `pci` sorted with eight unique actual PCI identities, and a reserved loopback `port` between 20000 and 60999

```sh
stage="$R/incoming/llava-clevr-5000-val16-${request%????????????????????}-stage"
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" start \
  --request "$request" --auth /approved/fresh.json --auth-sha "$auth_sha" --release /approved/release.json
```

The owner starts only within 55 seconds of the grant. Under the original shared lock it rechecks the exact prior Slurm terminal, dual parent and step views, and unused labels, then records private intent and runs a new 2CPU, 2048 MiB, zero-GRES pre step. After a second clear scheduler check, it attempts one exact 8CPU, 64000 MiB, 1DCU step with a 30-minute cap. The actual compute gate verifies numbered step, UID, cgroup, live TRES, boot, port and eight PCI identities with no KFD/FD holders. The GPU pre snapshot uses the same boot and monotonic time with a 65-second bound. The lock remains held through proof of the numbered step and is reacquired for an independent 2CPU, 2048 MiB, zero-GRES post. The GPU mount and native worker use the qualified .5006 layout with request-only private scratch. The exact scoped worker may be reaped at its deadline. Private scratch is retained for inspection.

`terminal --request "$request"` on the same staged script issues only read-only bounded queries. A complete result independently verifies original accounting, separate physical cleanup, and the sealed ledger hash and 32 outcomes under `R/results/llava-clevr-5000-val16-<request-prefix>/eval/once`. If a launch or post is unknown, retain its evidence and reconcile the original labels. The owner never automatically resubmits. The ledger and hardware result stay separate. `scoring_GO` requires terminal success, successful native paired inference across all 32 slots, and positive independent cleanup. Gold is never mounted.
