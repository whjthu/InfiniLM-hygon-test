# VGGT Freiburg room CPU preflight

This candidate qualifies the seven pinned room RGB-only pairs before model work. It calls the rendered backend's `_tensor_batch` and `_vggt_batch_content_sha256` with actual CPU Torch tensors and the upstream `extri_intri_to_pose_encoding` with the original camera fields. The receipt reports the native content digest, an independently assembled projection using the canonical hash helper, a repeated conversion, an order-sensitivity check, and the repeated native target digest. It records no raw images, tensors, timestamps, or camera values. The adapter requires the same seven windows and their fourteen verified pose associations. These are seven of twelve frozen windows from one scene.

## Exact inputs and runtime

Run `verify_local.py WORKSPACE PRIVATE_MANIFEST PRIVATE_LOCAL_RECEIPT` with two absent paths. It instruments the published reader to discover the exact ordered file set and verifies it locally. The generated private manifest records three public crop summaries and nine private files (three original association manifests, three crop arrays, and three pose JSON files). Keep the manifest and private inputs within the authorized evidence boundary. Stage those twelve members at their original relative paths inside the candidate data root. The reader does not require original BAG regions, original full-size RGB files, model weights, depth, point masks, or additional windows.

The source authority directory contains the existing T12 v2 `FRAMEWORK_FILES.json`, `UPSTREAM_FILES.json` and `DEPENDENCIES.json` at their pinned hashes. Use an isolated copy of the T12 framework identified by that manifest and verify the original backend before generating the published round-3 overlay. Replace the backend only in that copy and add the corresponding pinned reader. The preflight verifies all 323 authority framework entries with the one pinned backend substitution, the new reader, and all 69 upstream Python files. The staged Python package roots must contain only their expected Python sources and no bytecode cache. Bind the staged sources, authority files, and data root read-only. Keep the regular RGB-D deployment on its frozen original backend.

Use the previously qualified DTK26 container and explicit training runtime site path, with the expected Torch, NumPy and HIP versions from the pinned T12 dependency manifest. Supply absolute runtime site and pinned roots. Local validation confirms these authority files and the deterministic renderer, with target provider availability still to be checked. The prior zero-GRES nine-pair CPU result is operational precedent only. It used different RGB-D inputs.

## Target candidate envelope

One isolated attempt may be considered after fresh source/runtime checks, authorized private staging and live admission. The candidate is a single four-CPU step, 8 GiB, zero GRES, one task, a 120-second application epoch, and an outer timeout with a five-second termination grace. This package selects no node, launches no Slurm step, and transfers no files. The launcher must confirm the exact requested step has zero GRES. Reconcile the actual numeric step afterward against independent accounting for exit state, elapsed time and allocated TRES before declaring zero GRES. A missing receipt is an unknown outcome until reconciliation.

Under the qualified container, invoke the preflight with isolated Python and bytecode disabled. The existing DTK target environment must be initialized before the Python process starts. Disable network in the candidate environment and bind one writable evidence directory. The receipt name must be absent.

```sh
timeout --kill-after=5s 125s python3 -I -B -u /input-stage/preflight.py \
  --authority /input-authority --framework /input-framework \
  --upstream /vggt-source --runtime-site /training-runtime/site \
  --data-root /input-room --manifest /input-stage/minimum.json \
  --manifest-sha256 "$MANIFEST_SHA256" --receipt /evidence/cpu-preflight.json \
  --epoch "$CPU_EPOCH" --expected-job "$EXPECTED_PARENT" --expected-node "$EXPECTED_NODE"
```

The epoch, job ID and node come from separately authorized admission. The entry point checks job/node binding and rejects step-side GPU and DCU indicators before touching runtime inputs. It rejects populated GPU step IDs, nonzero step GPU counts and GPU or DCU task TRES. It records populated parent GPU variables as advisory, hides all accelerator devices and forbids Torch's lazy device initialization. A pass qualifies only CPU input conversion, order-sensitive native hashing and camera target encoding for the seven same-scene pairs. Model loading, optimizer updates, training readiness, GPU behavior and held-out effects are outside this preflight.
