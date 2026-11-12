# Target Z-Image runtime inspection

One CPU-only Slurm step on the assigned T07 node. Pin source before submission; same epoch for all budgets and terminal accounting. Check actual HIP Torch and required imports in the already-qualified offline training runtime. All GPU initialization, network connections, and tensor-file access are forbidden. Missing dependencies are recorded as prerequisites, not hidden by an overall inspection-success flag. No original tensor transfer, model construction, CUDA-policy reuse on HIP, GPU or performance claim.
