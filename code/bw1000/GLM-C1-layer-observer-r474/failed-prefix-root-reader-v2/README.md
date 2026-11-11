# GLM906 failed-run prefix audit

CPU-only reader for exact original 906/1121 failure. It reuses the pinned v1 request, frame, clock, scorer and independent post validators. The frozen helper is byte-identical to published v1.

The new reader validates only a contiguous completed prefix, all sixteen rank output hashes and both leader barrier digests. Missing or partial requests remain in the 33 planned / 32 screen denominators. Full-run acceptance and promotion are always false. Missing final driver sampler is recorded, not fabricated. All original inputs and detailed event windows remain private on R. No model load, Native/SDK execution, weight read, Slurm step or GPU work occurs.

Run check_failed_prefix_CPU.py for 19 malformed-evidence boundaries. Invoke read_failed_prefix.py with explicit --original, --source, --accounting, --plan, --watch and new --output. Output is private; publish only a bounded summary.
