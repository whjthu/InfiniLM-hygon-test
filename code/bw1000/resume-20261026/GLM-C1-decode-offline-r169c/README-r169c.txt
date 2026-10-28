Offline CPU only; no Torch/Native/Slurm/GPU imports or launches.
Root reviews/publishes/uploads then runs python3 -B offline_reaudit_r169c.py on original server.
Original 787 traces, Model runtime, original failed audit and watch are never overwritten.
Reader pins all16 original source SHA from safe metadata, replays old consumer failure, reparses locally.
Only audit AST edit is q=read_owner(p); all QA/lifecycle/DECODE ledger/coverage/overhead gates retained.
Legal repeated quant scopes paired in order via LIFO, all occurrences and fence markers retained.
Structural duplicate endpoints, missing/end-before-begin/crossed scopes are rejected.
Output: results/GLM-C1-decode-offline-r169c.json exclusively-created, compact aggregated medians/counts.
No raw events/native_events/native_calls, per-event timestamps, tokens or activations exported.
Intervals remain inclusive original-stream buckets, not GPU busy or exclusive TP cost.
No promotion or recovered performance qualification is automatic; root reviews actual result.
