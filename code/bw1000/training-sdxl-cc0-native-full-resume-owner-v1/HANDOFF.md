# SDXL R33 single-DCU continuous25 and CP20 recovery

The owner binds the exact frozen v3 controls for parent `1194900` and `f13r4n03`, including its original shared lock, protected legacy .189 check, one GPU submission and independent zero-GRES pre/post. The guest always runs the frozen HIP full checkpoint package with `--resume-step 20`. The science denominator is 25 continuous plus five new-process recovery updates. The original .2670 CP1 never enters the resume argument.

## Stage

Reuse the exact v3 owner, common owner, support directory, pilot, legacy proof helper, and three actual SDXL runtime receipts. Supply the frozen native full-resume package and a small independently checked root proof bound by SHA. The stage adds no Slurm calls.

```sh
/usr/local/python3.12/bin/python3 -I -S -B /approved/r33/host.py stage \
  --request "$request" --manifest-sha "$owner_manifest_sha" \
  --base-host /approved/v3/host.py --common /approved/llava-val16/host.py \
  --support /approved/llava-first2-owner --pilot /approved/pilot.py --legacy /approved/shared_standby.py \
  --runtime-import-proof /approved/n03-2666-runtime.json --tracker-proof /approved/n03-2667-tracker.json \
  --runtime-terminal-proof /approved/n03-2666-2667-terminal.json \
  --full-source /approved/sdxl-hip-full-resume \
  --runtime-proof /approved/r33-dtk-and-provider.json --runtime-proof-sha "$runtime_proof_sha"
```

The normalized root proof has only these sections. The original authenticated receipts should remain independently available.

- `rootfs` has `host_path` equal to the canonical production DTK sandbox path, `image_id` equal to `sha256:<64 hex>` from the independent rootfs identity, `method` as its lowercase evidence label, `source_receipt_SHA256` of the rootfs evidence, and `stat` as `[inode,size,mtime_ns,ctime_ns]` on the image directory. The stat is re-read on both HOST and compute to bind the path to the supplied image identity.
- `provider` has the same `image_id`, independent `receipt_SHA256`, exact `step`, `parent`, `node`, `accounting` in the standard nine-column owner order, `zero_gres` set to true, `hip_full_sources_SHA256` equal to the three actual derived source hashes, `native_cli_worker_count` equal to 0, and `native_full_checkpoint` set to true. The zero-GRES row must independently be terminal `COMPLETED 0:0` for n03 with no GRES and real Start/End.

Use the returned final `stage_SHA256` for the new ordinary v3-style grant. The intermediate `stage.json` proves inherited capture and mount dependencies. The final `continuity-stage.json` also binds image and full source and is required to start.

## Start and read back

```sh
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" start \
   --request "$request" --auth /approved/fresh.json --auth-sha "$auth_sha"
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" terminal --request "$request"
```

The namespace is `sdxl-cc0-full25-cp20-<request-prefix>` and the stage is its `-stage` sibling. Budgets remain 8 CPU, 64000 MiB, one DCU, and 30 minutes for compute. Independent pre and post each have 2 CPU, 2048 MiB and 0 GRES. The unique owner retains the private tmp marker, runtime bind, and exact single HIP ordinal preflight. The reader reports optimizer evidence separately from post cleanup. It independently reopens all four checkpoint manifests and members, verifies the recovery source and different process identity, and checks the two exact optimizer sequences. Any unknown original scope requires reconciliation of its own step and markers.
