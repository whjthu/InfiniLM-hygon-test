# SDXL CC0 first-update owner v3

The v2 request `fd93b793bbbd4203ad2839ec99a76b91` reached GPU step `1194900.2664`. The native trainer stopped at `import diffusers` before any forward or optimizer update. Its independent post `1194900.2665` passed. v3 uses the original read-only runtime mount and sets `PYTHONPATH=/training-runtime/site` after sourcing DTK, matching the preflight. The v2 explicit accounting parent and the earlier refusal to reuse the v1 scope remain in force.

This owner uses the original shared standby lock on parent `1194900`, node `f13r4n03`. Its only legacy exception is exact zero-GRES `1194900.189` with the original name and 1CPU/256MiB TRES, controller TIMEOUT, matching accounting, and absence from both queue and live statistics. It checks the original successful `.378` accounting. A numbered conflict on the node blocks the request. The owner sends no cancellation to another job or step.

## Stage

```sh
/usr/local/python3.12/bin/python3 -I -S -B /approved/host.py stage \
  --request "$request" --manifest-sha "$owner_manifest_sha" \
  --common /approved/llava-val16-owner/host.py \
  --support /approved/training-llava-clevr-first2-owner-v1 \
  --pilot /approved/fleet-dcu-standby-control-node-pilot-v2/pilot.py \
  --legacy /approved/fleet-dcu-standby-f13r4n03-v1/shared_standby.py \
  --runtime-import-proof /approved/n03-2666-runtime-import-transport.json \
  --tracker-proof /approved/n03-2667-actual-tracker-transport.json \
  --runtime-terminal-proof /approved/n03-independent-runtime-accounting.json
```

The unique stage is `R/incoming/sdxl-cc0-first1-v3-<request-prefix>-stage`. It pins and reads the actual .5007 compute and terminal receipts, 18-file pipeline manifest, published worker and witness, .5039 originals, and .5099 trainer and gencuda sources. It also requires the independent n03 runtime proof. The host stage scans small evidence and both CC0 inputs. The original worker rehashes actual weights in the GPU step before loading. Stage has zero Slurm effects.

The .2666 runtime import and trainer help proof is pinned at `01e28b94562367bff3fbb3fce25e563addb92df95b1592e45939e986f1f8d32e`. The .2667 Accelerator tracker proof is pinned at `5b140c07143d8bf0adfaa97056f8a9b5d52f3e5214c6026b0db6d8d3981c1a7e`. It records the actual TensorBoardTracker and two event files, preserving the existing trainer logging policy. The independent terminal receipt `bcf5d0fcbc9dc20682a0f63bd796ff865df7502feafe32e8c326b811d43617bb` confirms both originals completed at zero GRES.

## Authorize exactly once

The immutable root grant has exactly `request`, `parent`, `node`, `stage_SHA256`, `epoch`, `deadline`, `port`, `boot`, `pci`, `root_GO`, `ordinary_excluded`, and `fresh_physical_SHA256`. Set the latter to the actual independent root probe digest and bind its sorted eight PCI addresses, boot and reserved port. Set `root_GO` and `ordinary_excluded` to true only after independent checks. The original root physical probe `1194900.2659` used port 48373. Issue the grant on fresh readback, not just on that historical port. Use UTC server seconds with more than 300 and at most 2250 seconds total. The owner requires the grant age below 55 seconds at admission and checks an additional same-lock zero-GRES physical pre.

```sh
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" start \
  --request "$request" --auth /approved/fresh.json --auth-sha "$auth_sha"
```

The launch uses one 8 CPU/64000 MiB/1 DCU step with a 30-minute cap. Pre and post each use 2 CPU/2048 MiB, zero GRES and up to one minute. Exact compute checks bind numbered cgroup and controller TRES, same boot and eight PCI, no KFD/device FD and free loopback port. Inside the unchanged DTK image the process checks one actual HIP ordinal against the roster before calling the published worker. This owner allocates new private tmp/HOME and writable result directories with an exclusive marker. The original native trainer selects one of the two CC0 inputs for the single executed update. No scoring or throughput claim is issued.

```sh
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" terminal --request "$request"
```

The read-only terminal path reports actual optimizer update counts and independent cleanup separately. Success additionally requires digest readback of four checkpoint-1 members and the original device operand receipt. All output stays under `R/results/sdxl-cc0-first1-v3-<request-prefix>`. Lost transport or an unknown GPU attempt must be reconciled using those original role labels and intent paths. The script never automatically resubmits. Original .540 remains unqualified.
