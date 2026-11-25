# Qwen street-scene eight-DCU owner

The source packet leads from the original .1251 training-only data stage and the .1252 installed CPU sampler receipt to the published street-scene `native_entry.py`. It uses the original .220 framework and HIP assignment code, the original .1231 container and physical-snapshot pattern, and the original eight-DCU lease geometry of 64 CPU and 224000 MiB. The candidate has no selected node, parent or approved request. A scheduler-empty candidate with a prior GPU bridge in UNKNOWN fails the placement gate.

Version 2 binds the observed .1251 grid exactly as a one-row, three-integer matrix and flattens it only for comparison against the training batch. The 393 framework source pins are all required. The HOST may have the exact three enumerated Python 3.12 cache files found in the original readback. The HOST commands and all compute ranks use bytecode writes disabled and a per-request cache prefix that must not exist. Framework files and existing caches stay in place.

## Source and passive assets

Build the source-only ZIP with `python -B build_packet.py --out <private-absolute-path>`. Verify the exact packet and manifest SHA against the source claim and the actual ordinary publication. Stage through the existing authenticated HOST channel. The private archive contains only the source closure and four historical receipts. The original data and model stay at their existing paths.

After uploading the ZIP to an authorized private absolute HOST path, invoke its installer through the verified archive. The bootstrap reads the ZIP to check its transport hash before it imports any module.

```sh
/usr/local/python3.12/bin/python3 -I -B -c 'import hashlib, pathlib, runpy, sys; p=pathlib.Path(sys.argv[1]); assert p.resolve(strict=True)==p and p.is_file() and p.stat().st_size<=524288; assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]; sys.path.insert(0,str(p)); sys.argv=[str(p),"--packet",str(p),"--packet-sha",sys.argv[2],*sys.argv[3:]]; runpy.run_module("install",run_name="__main__")' "$PACKET" "$PACKET_SHA" --request "$REQUEST" --source-sha "$SOURCE_SHA"

/usr/local/python3.12/bin/python3 -E -s -B -X "pycache_prefix=$RUN/never-populated-python-cache" "$RUN/src/owner.py" assets --run "$RUN" --source-sha "$SOURCE_SHA"
```

`RUN` is the exact new project results path `qwen-textvqa-first-two-$REQUEST`. The first call installs only named source files and independently reads them back. The second hashes the 14 original model files, original LoRA initializer, 393 original framework files, staged recipe, vision assets and fixed passive runtime source. It independently rechecks the existing training archive installation and .1251 report. It emits the asset receipt SHA. The large hash pass is done before the placement decision. Dispatch later compares the same HOST and all file stat identities, then checks the exact loaded Torch, Swift, Transformers, Accelerate and vision providers inside the admitted GPU process.

## One admitted run

The private scope contains `request`, `parent`, `node`, actual `publication`, `port`, `source_manifest_SHA256`, `DCU=8`, `CPU=64`, `memory_MiB=224000`, `whole_s=1720`, `GPU_GO=true`, `exclusive_node=true`, `replay_allowed=false`. The private authority matches the same four request/parent/node/publication fields. It contains `coordinator_evidence_SHA256`, `exclusive_no_other_dispatch=true`, `ordinary_reservation_reconciled=true`, a positive verified `legacy_bridge_state`, `observed_epoch`, and `expires_epoch`. The coordinator independently verifies the proof and holds the node through the whole epoch. A new start requires the actual proof file and both exact private digests. Source staging and passive asset checks do not authorize placement.

```sh
/usr/local/python3.12/bin/python3 -E -s -B -X "pycache_prefix=$RUN/never-populated-python-cache" "$RUN/src/owner.py" detach --run "$RUN" --source-sha "$SOURCE_SHA" --scope "$SCOPE" --scope-sha "$SCOPE_SHA" --authority "$AUTHORITY" --authority-sha "$AUTHORITY_SHA" --coordinator-proof "$COORDINATOR_RECEIPT" --assets-sha "$ASSETS_SHA"

/usr/local/python3.12/bin/python3 -E -s -B -X "pycache_prefix=$RUN/never-populated-python-cache" "$RUN/src/owner.py" reconcile --run "$RUN" --source-sha "$SOURCE_SHA"
```

The receiver records a keeper intent and starts one detached HOST owner under the absolute timeout binary. It returns the keeper PID, startticks and bounded epoch, with exclusive stdout and stderr files. The HOST owner checks both scheduler membership and occupancy views, checks parent-local CPU and memory, records intent, and uses the original zero-GRES physical preflight. Only then does it attempt one GPU step. The whole owner stays within the original 1720-second epoch. Native training has a 1320-second cutoff from the same epoch. When a GPU terminal becomes visible, a separate zero-GRES post step checks the same boot and eight PCI devices, KFD and same-UID FD cleanup, the port, and the exact owned process identities. Read-only reconciliation requires authoritative current nine-column accounting and the numeric pre/post witnesses. It independently reads rank receipts and keeps training qualification separate from physical release. UNKNOWN retains the reservation and requires reconciliation. Blind or held-out evaluation is a separate workload.

Run `python -B -X utf8 check_local.py` from this directory to rebuild and independently install the real packet in a fixture and capture the actual query, launch and reconcile entry paths. No cluster process is needed.
