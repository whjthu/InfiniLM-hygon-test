# HunyuanVideo .4560 full-base CPU adapter reload, inventory V3

This scope independently constructs the full original transformer in the original DTK runtime and attaches the exact .4560 LoRA. It performs no forward, sampling, training, optimizer construction or adapter export.

The original target adapter is `R/results/training-hunyuanvideo-t09-abo-domain-n14-fd-v3/outputs/hunyuanvideo-1.5/train/training-hunyuanvideo-t09-abo-domain-n14-fd-v3/adapter-final.safetensors`. The worker accepts exactly 3,570,312 bytes with SHA-256 `2436d2a23b0a9415f9f8893e261008846afe37b8185dade67ce7b1fc65e02b76`. The current original transformer is bound to the historical complete 33,306,632,192-byte SHA and exact compute file identity. It does not perform a second whole-file hash. Widths in this scope refer to the transformer and its buffers only.

The worker verifies the installed V3 source, exact private 393 framework, 38 native files, actual DTK runtime pin and original single compatibility bind. It reuses the original checkpoint wrappers and adapter digest helpers from backend SHA `2331d0fa942eb3908dc5392df6008355524f013a0fbdb5067202f9a74400ccae`. The original LoRA registration has rank 4, alpha 8, dropout 0, targets img_attn_q and img_attn_v, no bias, FEATURE_EXTRACTION and adapter name default. The exact 216 trainable names, shapes, CPU BF16 dtypes and 1,769,472 elements must match. Each value is copied and verified, followed by the original named SHA `bd9dfa706ee5dca1d4732caebada76d0cdb0a14094d5487acee52634d3b60221`.

The CPU allocation is 8 CPUs and 64,000 MiB with zero GRES for up to 480 seconds on `1103606/f13r3n14`. The original safetensors has 1,793 F32 entries and 8,326,608,160 elements. The exact 199,544-byte header body (after the 8-byte length prefix) has SHA-256 `a6feb5932d0522fb8824f66eebef5569b0f3072cfb2cc2a95ad921280d62e0c2`. It binds the complete BF16 materialization at 16,653,216,320 bytes. After native low-memory loading the header keys and shapes must correspond one-to-one to the registered parameters and persistent buffers in `state_dict(keep_vars=True)`. Nonpersistent buffers are counted separately. After LoRA injection and original checkpoint wrapping, the original objects must remain frozen in exactly the same CPU BF16 storage. Bounded page-stride reads and the previous RSS guard apply against that exact inventory. The actual V2 .4566 failure remains separate.

Scratch is unique to this request. Original assets and model are mounted read-only. All three GPU visibility masks are `-1`, and the worker rejects device descriptors and Torch lazy GPU initialization.

V2 accepts exactly three unique TRES fields in either order: `cpu=8`, `node=1`, and either `mem=64000M` or `mem=62.50G`. The original .4565 accounting actually used `cpu=8,mem=62.50G,node=1` and ended FAILED 1:0 before the worker result. Its compound compute-gate error does not identify whether the live controller used the identical memory representation. V2 records the live controller string and the two unchanged environment predicate results separately from the raw terminal accounting string.

## Once-only entry

```text
python3 -B host.py install --request <new-32-hex> --publication <new-published-commit> --manifest <source-manifest-SHA256>
python3 -B <R>/incoming/t09-abo-4560-full-cpu-reload-inventory-v3-<new-32-hex>/src/host.py start --request <new-32-hex> --publication <new-published-commit> --manifest <source-manifest-SHA256> --stage-sha <stage SHA256> --authority <canonical-private-root-authority> --authority-sha <sha>
python3 -B <R>/incoming/t09-abo-4560-full-cpu-reload-inventory-v3-<new-32-hex>/src/host.py terminal --request <new-32-hex>
```

`authority` contains `request,parent,node,stage_SHA256,manifest,publication,root_GO:true,CPU_GO:true,GPU_GO:false,zero_GRES_once:true,allowed_standby:"1103606.4564",observed_epoch,expires_epoch`. Root verifies the current authorized observation separately. The lifetime is at most 65 seconds. The owner rechecks exact original GPU and post terminal rows, both parent views, current .4564 in scontrol and independent accounting, and any other step in squeue or scontrol. It subtracts the exact existing standby's 8 CPUs and 16 GiB from the owned node allocation. No signal or replacement is issued. Every fresh start has one exclusive intent and one srun opportunity. Reconcile any unknown by its original accounting label and work directory. The separately executable terminal reader requires the actual distinct numbered zero-GRES row and matching stored worker output before approval.

Local validation runs the exact original digest helper against a small tensor fixture and checks header duplicates, payload coverage, registration and state mismatches, nonpersistent buffers, original parameter object/storage continuity, named adapter mappings, and unchanged TRES admission. Actual local model loads are zero.
