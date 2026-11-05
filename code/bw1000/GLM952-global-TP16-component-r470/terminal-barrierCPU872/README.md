# CPU872 same-marker control diagnosis

`910371.872` COMPLETED0/33s, 16 owner processes / 256 original waits / 2 normal workers. CPU-only, no SDK/Runtime/GPU. Same published `device-v1/shared.py` SHA4c145… called directly; original870 files untouched.

Usage: `python -B read_original.py /absolute/path/GLM470-barrierCPU872-original-sealed-v1.json.gz --output /absolute/path/audit.json`.

Node1 ranks8..15 waited30.0434..30.4456s in original wait(created00); node0 all16wait sums0.0032..0.2485s. This includes visibility/retries/read/scheduling, without errno/cache trace. Current success rejects deterministic namespace/rank/epoch mismatch; it does not prove original870 historical FS state or CCL health. No repeated device execution. The next proposed control uses qualified one-shot two-leader framing plus local pipes; CPU producer/consumer proof precedes any different device claim.
