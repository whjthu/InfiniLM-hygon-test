# GLM-5.3 TP16 paired Slurm launcher

Install these HOST files beside the exact 16 runtime files and SOURCE_MANIFEST.json from full-model-paired-worker-v1. Mount the combined directory read-only in the container. Supply a separately published GPU_RESOURCE_CLAIM.json and its digest before invoking host_launch.py with the 32-hex run ID. The claim binds the run ID, both source manifests, both Native hashes, parent 1103606, nodes f13r1n05/f13r1n07, one run per arm, and the 16-card resource and deadline fields consumed by main(). This package contains no executable claim.

The launcher reserves the run ID exclusively, then executes baseline followed by single-drain on the same pair. Each arm requires a fresh CPU-only precheck and leased PCI/KFD gate. The next arm requires terminal accounting and a clean CPU-only postcheck. Missing accounting keeps the reservation. Read the original receipts to reconcile an interruption. Each arm has an independent epoch, output, and Slurm labels. This fixed process order leaves order and time confounded and is suitable for an engineering screen.

The HOST rehearsal uses local fixtures. Target Apptainer bind paths, Slurm accounting visibility, hardware checks and whole-model performance still require live verification.
