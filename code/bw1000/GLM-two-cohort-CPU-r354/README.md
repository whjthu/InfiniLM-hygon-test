This private CPU card consumes the unchanged GLM Native038d CAPI with two
independent persistent rows (C2), one row per forward (M1), TP8/PP4/32 owners.
It is not an executable GPU owner, constructor qualification, or HTTP service.
No current307/349 source is modified.

`decode_wave_r9225.py` is copied byte-for-byte from the existing719 scheduler.
`consumer.py` supplies current row/past/token mapping, EOS retirement, a finite
pair budget, round-boundary cancellation and reset/close callbacks. `native_binding.py`
executes the exact current forward argument layout and supplies C2/32768 factory
arguments without calling that factory. Existing owner-thread control, loader,
preweight checks, Native drain and cleanup must remain in the eventual owner.

Run `check_CPU.py` with the fixed project Python. It executes serial and deferred
rounds with32 real CPU owner threads, four PP stages, bounded adjacent-stage
mailboxes and WORLD feedback barriers. It also records the real private binding's
factory arguments and forward calls against mock Native/Torch boundaries for all
32 owners. Tests cover independent past lengths, greedy synthetic logits, EOS
retirement, normal reset/close, agreed cancellation and failure cleanup without
retry. These are CPU protocol checks, not model arithmetic or GPU CCL tests.

The only next system candidate is C2M1 on32 cards. Prefill both independently with
the existing fixed bounded prefill policy, then compare serial F0/E0/F1/E1 with
deferred F0/F1/E0/E1 in balanced ordinary ABBA using the same two frozen complete
requests, one loaded context, source/math/precision and rows2 allocation in BOTH
arms. No reset occurs between the two live rows; no row refill or reuse until both
finish. The remaining single row after EOS automatically uses width1. Actual
mid-forward Native/CCL failure is fail-stop, never a frontend ledger replay.

Required future preparation is a private full owner adapter and exact fixture/
budget/claim, after349 terminal. Keep all existing preweight checks (including the
4GiB floor), original deadlines, finite logits and original normal32-owner destroy.
Measure constructor/load/finish/prefill/decode memory rather than assuming the
source-derived C2 cache estimate proves fit. No GPU is authorized by this card.

Any future timing must report per-request emitter-local first/last token intervals
and common-coordinator total useful throughput over all32 service cards, with
TTFT/ITL/EOS/unfinished requests. Compare same-owner complete schedule wall and
aligned max32-owner trial scope. Never subtract unrelated node clocks, sum
independent per-node rates, or apply old719's PP2/M16 speedup to PP4/M1. Effect on
single-user latency and Model wall remains UNKNOWN until actual qualification.
