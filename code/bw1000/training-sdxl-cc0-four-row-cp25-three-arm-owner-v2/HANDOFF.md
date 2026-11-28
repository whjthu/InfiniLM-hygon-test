# SDXL four-row CP25 six-slot owner

The owner retains the frozen n03 single-DCU controller, shared standby lock, protected .189 check, physical pre/post, deadlines and resource limits. Stage uses actual .2769 continuous CP25 and .2762 sealed originals in place.

## Stage and verify

```sh
R=/public/home/aclsylqidf/hygon-inference-refactor-20261014
PY=/usr/local/python3.12/bin/python3
"$PY" -I -S -B /approved/four-row-pair-owner/host.py stage \
  --request "$request" --host-sha "$host_sha" \
  --runner /approved/four-row-cp25-three-arm-v2 \
  --r33-proof /approved/specialized-sdxl-2675-native-full25-cp20-exact-recovery-and-checkpoints.json \
  --training-proof /approved/specialized-sdxl-2769-four-native-rows-25-plus-5-exact-CP20-and-clean.json

prefix=$(printf '%.12s' "$request")
stage="$R/incoming/sdxl-cc0-four-cp25-six-ev2-$prefix-stage"
"$PY" -I -S -B "$stage/host.py" verify-plan --request "$request" --stage-sha "$stage_sha"
"$PY" -I -S -B "$stage/host.py" verify-terminal --request "$request" --stage-sha "$stage_sha"
```

Use a fresh 32-hex request and the stage receipt SHA for both commands. The original staged .2675 dependencies remain at their canonical paths. The frozen science manifest SHA-256 is 51e38cf2f8d66926fceeec51ed1f23e59e592895b005d2a286223d07e3599612. Stage consumes no Slurm step.

Under the shared lock, verify-plan reruns the original scheduler guard, makes one exclusive intent and invokes one exact 2 CPU, 2048 MiB, 0 GRES step with a two-minute cap. Compute validates actual UID, node, cgroup and the live controller record. In the frozen DTK image, the stdlib native verify-plan reads the exact checkpoint manifest and adapter plus the original ledger and six BF16 noises. All material binds are read-only, there is no ROCm bind, and device visibility is explicitly disabled. If accounting lags or the result is unknown, use only verify-terminal to reconcile the same label and do not replay.

## Independently authorize GPU

The fresh ordinary root grant fields remain request, parent, node, stage_SHA256, epoch, deadline, port, boot, pci, root_GO, ordinary_excluded, fresh_physical_SHA256. Bind an independently checked clean placement and the exact staged request. Verify-plan establishes data access and identity. Authorization remains a separate root decision.

```sh
"$PY" -I -S -B "$stage/host.py" start --request "$request" --auth /approved/fresh.json --auth-sha "$auth_sha"
"$PY" -I -S -B "$stage/host.py" terminal --request "$request"
```

Only after independent CPU terminal evidence passes, the original controller performs its separate zero-GRES physical pre, one 8 CPU, 64000 MiB, 1 DCU step with a 30-minute cap, and independent zero-GRES post. New private files are written under R/results/sdxl-cc0-four-cp25-six-ev2-<prefix>/durable/four-row-pair-once. The original .2762 images and .2769 checkpoint mounts are read-only. The terminal reader reports the six-slot outcome separately from physical cleanup. The science comparison.py can reverify the original twelve and new images with its separate certificate. This owner never opens an existing anonymous review.

The local test checks pinned original sources and archive hashes, shared-lock and no-replay wiring, exact resource and read-only bind argv, live and terminal checks, and rejection of wrong scope or nonzero GRES. Scheduler and container calls in the test are fixtures.

This successor also rejects the original .2774 request. The archived .2803
evidence fixes the seven-member normalized Euler config. The worker seals the
current raw config and per-slot before/after hashes. HOST terminal and the
science composer independently recompute the historical reconstruction and
check the raw stability receipts. Zero-GRES verify-plan remains the original
material preflight and does not generate, refine the bridge, or authorize GPU.
