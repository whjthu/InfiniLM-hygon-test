# Gemma CP20 first-forward diagnostic

The original .851/.860 artifacts and jobs remain unchanged. The exact public .860 outcome is e385031e45863ce7ae1849150f47edd523181099544891017f4f0050f363fc67. The independent .890 readback localized its first mismatch to rank 0 at global step 20, microbatch 1, before the original optimizer update 21. Historical first-forward RNG/layout was not captured and cannot be reconstructed from saved endpoint RNG.

The source delta connects one observer to the continuous before/after methods and to the corresponding resume methods. It records only rank 0 and control rank 1 at that coordinate. It hashes Python, NumPy, CPU and the selected local HIP RNG independently, reuses the exact prior aggregate training-context projection and tensor header helper, and reuses the original input hash and loss hash/scalar. No payload copied by the new observer enters the report. Both ranks write one exclusive, bounded result. The other six ranks do no new RNG, context, layout, storage or tensor reads. The original witness still performs its original restore and input checks on all ranks.

The resume-only plan preserves the original 800-row view, 8 HIP ranks, batch1/GA4, original seed and native skip 80, all original AdamW/LoRA/BF16 arguments and max_steps=25. The callback stops normally at the end of update 21 after four real forwards per rank, with no checkpoint save at that boundary and no full forward-entry capture. The continuous callback provides the same schema if a separate prospective run is authorized. It does not make old continuous observations available and is not dispatched by this plan.

## Render locally

```sh
python -B prepare.py overlay \
  --original /approved/training-gemma-usgs72-fresh-initial-domain25-v1/overlay \
  --context /approved/training-minicpm-forward-entry-state-CPU-v1/framework/vlm/runtime/forward_context_state.py \
  --output /private/new-overlay

python -B prepare.py plan \
  --original-plan /approved/training-gemma-usgs72-fresh-initial-domain25-v1/resumed-plan.json \
  --request "$request" --port "$fresh_port" > /private/new-plan.json
```

The generated JSON is executable argv for the existing unified Swift launcher. The only argv replacements are the private framework Swift entrypoint, private external callback and private output directory. It gives the precise future /results scope with training_ready=false. It does not by itself launch Slurm or confer an allocation.

## Target dependencies and entry

Before adoption, root must confirm the original .851 complete independent terminal and its exact fourteen CP20 pins, the unchanged 800-row data/model/runtime, the production Trainer full SHA and native skip function, plus the current eight-device ordinary admission, original shared lock and independent post. Install the previously verified original framework view at the new plan’s framework path with the five exact overlay members applied into that private copy. Verify that the actual unified Swift module path points to this private framework and that the checkpoint and dataset remain original read-only paths. The old stage/worker hardcode the old result namespace, request and port. They cannot be used directly for this new request.

The future approved owner needs to supply the private, SHA-bound NATIVE_RESUME_CONFIG file with case=T02.sft, dataset_SHA256=74fd9aad043441b7e29578223605a0030a93c5c9614057707b381175cda9a0cc, original structural binding path, checkpoint_step=20, original CP20 path, all fourteen checkpoint_files pins and actual trainer_SHA256. It must pass NATIVE_RESUME_CONFIG, NATIVE_RESUME_CONFIG_SHA256, NATIVE_RESUME_EPOCH, NATIVE_RESUME_DEADLINE and NATIVE_RESUME_MAX_SECONDS, with the same bounded epoch and independent cleanup reserve. With those facts verified, the approved owner executes its existing Python with -B plus the plan's command under the plan's environment_overrides, MASTER_ADDR=127.0.0.1 and private request-only scratch. It must retain exclusivity, fresh boot/PCI/KFD/ports and GPU-cgroup ownership. No launch is authorized by this source package.

Expected private evidence is resume-witness/rank-00/first-tail-forward.json and rank-01/first-tail-forward.json, alongside the eight existing resume witness results, one actual optimizer update and 32 total forwards. Compare the two captures as rank-specific evidence. A within-rank before/after change is an observation, not a diagnosis by itself. This is an engineering diagnostic with no production delivery or performance qualification.

Run python -B test_local.py for generated callback and stop-control checks, true Python/NumPy state readback and fixture CPU/HIP digest accessors. The tests do not import Torch or access an accelerator.
