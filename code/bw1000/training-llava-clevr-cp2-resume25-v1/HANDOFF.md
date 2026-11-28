# LLaVA .4983 CP2 continuation CPU preflight

This scope binds the original 17 CP2 members from the published .4983/.4984 archive. The original checkpoint remains mounted in place. The program hashes its members without deserializing them, then parses only pinned trainer state. The observed cursor must be step 2, epoch 1 and nominal max steps 25. Eight local microbatches per rank and accumulation four give two updates per complete epoch. The next native loader starts epoch 1 without within-epoch skipping.

Use the same original Swift provider under `/swift-runtime/site`, plus the authorized OneVision processor-only metadata directory. Bind the original train-only CLEVR64 and 64 images at `/llava-train-only`. Mount a private read-only source mirror at `/llava-cpu-authority` with the existing public archive `4983.json`, original first-two `original-launch.json`, and existing `.4719 INPUTS.json` / `.4878 TENSORS.json` identities. Do not copy CP2 or training data into the source package. Reuse the actual installed processor-side runtime. In the CPU guest, retain BF16 CLI fields, offline mode, four production worker fields and CPU-only diagnostic traversal. The diagnostic actually traverses with zero DataLoader workers inside eight Gloo rank processes, and never calls model forwards.

In a separately authorized zero-GRES step, with a new mode-0700 output location and request-private node-local scratch, run:

```sh
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export GLOO_SOCKET_IFNAME=lo PYTHONPATH=/swift-runtime/site
/usr/bin/python3 -B /llava-resume/cpu_order.py prepare --output "$OUT" --checkpoint "$ORIGINAL_CP2"
/usr/bin/python3 -B -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --master-addr 127.0.0.1 \
  --master-port "$CPU_PORT" --log-dir "$CPU_LOG" /llava-resume/cpu_order.py rank \
  --output "$OUT" --processor "$PROCESSOR_ONLY" --scratch "$PRIVATE_TMP" --rendezvous "$PRIVATE_SHM/store"
/usr/bin/python3 -B /llava-resume/cpu_order.py merge --output "$OUT"
```

The parent must verify an actual node-local tmpfs for the common `/dev/shm` directory and allocate it exclusively per request. Its `store` path must initially be absent. Preserve rank failures and independently establish that every child has exited before clearing only the request-owned rendezvous. The standard CPU outer owner must enforce bounded duration and verify an actual completed zero-GRES numeric step. Each rank invokes deployed `SwiftSft._get_dataset`, checks both full Arrow tables against the exact CLEVR source, and traverses the live Swift `DataLoaderShard` using the verified logical-to-physical projection. No GQA roster or plan is consumed.

The CPU output is a private plan, covering update slots 3–25, with update 3 beginning at native rank offset zero. The final half of epoch 12 remains outside the training budget. A fresh eight-DCU production owner must later bind exact target CPU output, actual CP2 and real restore hooks for adapter, AdamW, scheduler and all eight device RNG states, and verify actual production four-worker ordering. The production guard consumes actual first-step loss, optimizer counters, LR, parameter delta and forward data, then requires all 736 real forwards and 23 real updates plus native CP10/20/25. The CPU check cannot establish real restore, training, independent clean release or equivalence to an uninterrupted run.
