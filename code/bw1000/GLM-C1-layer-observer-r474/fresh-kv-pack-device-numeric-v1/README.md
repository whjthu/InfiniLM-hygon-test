# GLM474 fresh K/V pack device numerical gate

The worker executes both fresh K/V pack exports from the Native library built in Slurm step `1103606.918`. The device inputs use the production TP16 T1 sizes and BF16 layout. The CPU reference uses the production fallback's exact copy ranges. Each head's first 192 K words come from that head's projection, the final 64 come from the shared rotary row, and all 256 V words come from the projection after its first 192 words.

The numerics gate covers all four seed and tile cases and 36 explicit argument rejections. It checks device canaries, bitwise output, read-only input payloads, rejection return codes and unchanged output after rejection. Each attempted case and successful comparison is persisted. A killed worker leaves the preregistered denominator and the last recorded attempt in the private JSONL file.

The inputs are controlled component fixtures. This scope ends at the executed pack C ABI. The Tensor caller's AR selector, the upstream FP8 projection, downstream attention, full checkpoint and latency require separate validation.

## Target launch integration

- Use one newly admitted `f12r2n13` DCU from parent `1103606`, four thread-bound CPUs, and 16384 MiB. Check unique request/label and original namespace before reserving; keep the reservation if delivery or accounting becomes unknown.
- Stage this directory and the frozen referenced files under the checkout exposed as `/source`. Bind the Refactor root read-only at `/refactor`, the private original-run namespace read-write at `/scope`, and the existing `/opt/hyhal` read-only. Use the `dtk26-sandbox-r1` Apptainer image with `--rocm --no-home --pwd /`. The native worker requires the exact original 71 provider paths and SHA256 values under this mount. The wrapper restores the Slurm GPU visibility masks after sourcing the original CPU environment.
- Obtain `/scope/before.json` from the established zero-GRES HOST `snapshot(False)` on the actual target node, with its module-local `NODE` and `STEP` set to the claimed node and parent. Use the same boot and fresh monotonic reading, eight unique PCI identities, zero holders, plus independently verified parent step accounting and work ownership. The worker checks these snapshot fields and resolves the actual assigned card through HIP and KFD.
- Write `/scope/intent.json` once with `label`, `request`, `parent`, `node`, `archive_SHA256`, `pre_admission_SHA256` for the exact before bytes, `replay_allowed: false` and a positive UTC `epoch`. Derive `deadline` and `work_deadline` from the same epoch using `+300` and `+270`. The target wrapper bounds the child with the original work deadline and five-second kill grace. Do not reset the epoch while waiting for a step.
- After root-side publication and admission, invoke the staged worker through `/bin/bash /source/code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-device-numeric-v1/device_inside.sh`. `/scope/probe.jsonl` must be absent before execution. Preserve the entire attempt and independently reconcile Slurm terminal accounting, unchanged whole-node PCI identities, cleared KFD/FD uses and original worker PID/start ticks before releasing the node.

Existing checkout references are the actual .918 archive, CPU308 PLAN and ENV, routed I128 typed HIP/guard helper, and the TP16 assigned KFD helper. The worker checks the relevant source digests before use.

## Local checks

Run `python -I -S -B check_local.py -v` from the staged package, located within the checkout with its reference files intact. The CPU fixture independently computes all BF16 words through a scalar indexed path, exercises the original caller's complete denominator, and rejects corrupt output, guards, faulty rejection and bad admissions. `bash -n device_inside.sh` checks the wrapper syntax. HIP execution, Slurm placement, target loader verification and cleanup remain target evidence.
