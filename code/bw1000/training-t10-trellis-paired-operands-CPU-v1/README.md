# Paired original A100/HIP LoRA operands on CPU

Use authenticated original109 Hygon payload7fad7cd0 (28,919,769 bytes) and new originalA100 firstbackward e1131c8d (28,474,215 bytes). Four fixed original rank2 LoRA sites, unchanged real cached inputs/initial/model. No new model or GPU math.

The actual analysis module is committed in the unified framework. Three CPU algebra/invalid-input tests precede real operands. Compare both native gradients, CPU fullFP64 oracles, and both cross substitutions. Decompose deltaG into input, upstream and interaction terms, retaining their Gram cross terms; no misleading sum of independent percentages. Fulladapter, optimizer/model quality, recovery and performance remain separate.

One Hygon epoch starts before source install and binary input stream, whole300/body180/Slurm240/outer240 seconds; TERM5/KILL5. T10 companion a14r2n02,4CPU/8GiB/0GPU. Framed SSH stdin carries source header then exact28MB private binary; no raw tensors in Git or tool output. A100 download is a separately timed bounded preparation read; no source replay on unknown outcome. Fresh unique scope/node/parent/label checks, exact sources and both payload SHA before torch.load(weights_only=True,map_location=cpu). The original109 and A100 GPU scopes are never repeated.
