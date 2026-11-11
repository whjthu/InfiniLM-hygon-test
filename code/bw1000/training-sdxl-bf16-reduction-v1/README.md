# SDXL BF16 reduction policy diagnosis

One candidate on the original T06 node and fixed input. The already closed default and math-SDPA arms failed the unchanged5% update gate; original CPU diagnosis localized99% of squared update error to27690 opposite-sign gradient elements. This candidate changes only the native allow_bf16_reduced_precision_reduction flag from True to False. It records policy readback and actual SDPA/ATen operator counts; no actual accumulator precision is inferred from a flag alone.

The original native BF16 model/LoRA,initial state,inputs,loss,AdamW and comparator remain byte-identical. No positive result authorizes25-step/performance/effect promotion. If the native flag is alreadyFalse,the CPU policy check stops without GPU execution. This is a numerical diagnosis,not paired performance; the earlier closed reference is reused,not replayed.

One original1200second epoch covers installation,CPU policy,unique GPU dispatch,cleanup and independent CPU comparison. Native stops at epoch+660,GPU cap at epoch+780. Slurm caps shrink from the original epoch. Qualified bounded terminal accounting helper is copied unchanged. Fresh node8PCI/KFD gate,one visible GPU,independent post; record all missing/RUNNING/failed accounting without replay. Source subset5files from the unified317file framework is verified; do not claim a full317file target installation.
