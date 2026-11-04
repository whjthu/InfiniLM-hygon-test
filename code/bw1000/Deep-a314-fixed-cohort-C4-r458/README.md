Private explicit a314 fixed-cohort C4 consumer. The ordinary rows1 public
provider and its current runs are unchanged. No Native/SDK or mathematical TU
is rebuilt. This package does not start HTTP or create an online scheduler.

The typed profile is rows4, **32768 per row**, TP8PP1, HC0/r3660/fullpre0. Both
serial-C1 and true-C4 arms use the same rows4 resident engine. `entry.prepare`
consumes the actual 50-object a314 binding and CPU945's ordinals 0,1,128,129:
first two GSM (8 train shots), first two MMLU (5 dev shots), complete messages,
their own Deep IDs, original 4096 budget. It never re-encodes or selects answers.
Raw bodies, IDs and gold remain private; CPU output retains a private bound packet.

`entry.execute` composes the original loader/owner Context and forward with the
new continuation. Prefill is sequential by row in chunks of 1024. Every final
prefill sample is emitted at its actual return time, before the next row starts.
C4 decode passes active row IDs with independent past/offsets/positions, returns
count*vocab BF16 plus count int64 IDs, and retires EOS rows. No slot is reused.
Only the complete cohort or an error/cancellation drains and resets all rows.
Normal same-thread Context destruction remains inherited.

The proposed schedule is two warm cohorts then ABBA formal, each cohort four
requests, 24 logical requests total; 4096 is a cap, not a forced output length.
No device claim is activated. Full32K rows4 persistent memory and original
8GiB preweight /16GiB postload floor remain UNKNOWN. Existing C1 memory
observations and row-scaled cache source are insufficient to prove peak fit.
Future exact scope must include all8cards, reset/prefill/decode/D2H/host/drain,
full per-user times and cap/EOS/quality outcomes. No online SLO or PD gain claim.

`run_CPU.py` is the only currently authorized execution: one4core8GiB<=300s
CPU job, no Native/SDK/Runtime/weight/Model/GPU. It invokes actual new prepare,
file handoff, Context/forward/continuation; Native/libffi, weights and memory
leaves are explicit recordings. It does not replay945 encoding or old suites.
