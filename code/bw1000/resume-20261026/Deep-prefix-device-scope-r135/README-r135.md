# Deep prefix raw-device query fix

Failed actual step 910374.1131 is retained as failed. Owner thread assertions passed; raw cudaGetDevice queried the thread default rather than establishing the Runtime device. Runtime APIs can use temporary device guards, so owner identity does not imply a persistent current-device binding.

This candidate binds the Runtime device only around raw inspection and cudaMemGetInfo operations, strictly checks the resulting device, and restores the prior thread device on normal and exceptional returns. Rank, Runtime owner, 43-layer state, finite/numerical, all16 prefix cases, 4 GiB reserve, cleanup, and no-F64 requirements remain.

CPU_build_r135.py compiles one NativeSparse HOST TU, reuses every other exact object from qualified CPU453, verifies fresh dependency closure, unchanged old object hashes, all old exports, Bsymbolic and ldd-r. A separate four-case CPU mock tests binding/restoration; it is not GPU evidence.

Before any weights or model construction, GPU_run_r135.py runs an actual8 Runtime-owner-thread device contract with deliberate previous devices, normal/exception restoration, and memory reserve. Failure stops before loading weights. Success proceeds to one model load and unchanged16 actual43-layer prefix-state cases. No MTP head, QA, service or TPS claim.

Root sequence: stage source manifest -> CPU-only launch_CPU_r135.py -> verify actual completed CPU receipt and DSO -> publish sources/results on own experimental branch -> create actual GO from false template with exact manifest, CPU receipt, DSO and pushed commit -> launch_CPU_keeper_r135.py once. Never use inherited archival r129 manifest as a launch gate. No GPU is authorized by this package itself.
