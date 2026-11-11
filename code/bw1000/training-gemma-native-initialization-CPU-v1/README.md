# Native initial trainable CPU qualification

Reconstruct exactly316 shared framework files from the closed original Gemma314-file source plus five published changes. Verify every byte and executable mode before one CPU-only PEFT/AdamW and callback test session. Four affinity CPUs, two Torch threads,300seconds one epoch; no Slurm allocation or visible accelerator.

The callback fixture substitutes only the separately qualified native model-contract validator and uses tiny synthetic PEFT tensors. Full source Gemma initial.pt is not read or transferred. This does not establish production parameter restoration, matching RNG/dropout, GPU math, recovery, quality or throughput. The original Gemma weight relay continues independently without replay or changed deadline.
