# PD445 mixed-demand queued performance card

This private entry reuses the exact Native/52-object state codec used by836.
Native/SDK/weights/greedy policy, all8 owner masks, C1/TP8/PP1/context32768,
whole-state transfer, generation poison, reset and16GiB admission floor remain.
The candidate is a workload/scheduling measurement, not a new model optimization.

Changed: per-request complete prepared fixture/prompt identity/past/quality/budget;
6 warm requests and one balanced four-cohort ABBA (16 formal) using two fixed
LSSL/SLLS orders. Two co replicas each process one L and one S in every formal
cohort. All arrivals within a cohort are simultaneous; queues are included.
C4 means4 external queued requests, never4 active rows or decode batching.
The original scheduler still holds P's export snapshot until D can receive and
ACK the entire state; that backpressure and head-of-line delay are measured.

L is the original complete8489-token tool request, output cap128. S is original
MMLU30 request bytes (SHA33d2f8...), observed150 tokens and312 generated including
EOS in qualified1221, now cap1024. No cropped prompt, padding, repeated forced
output, hidden cap or ignored EOS. Old1221 is provenance only: this exact Native
and new short state path still need actual gold/EOS. MMLU18/01 were excluded
because their original answers were wrong. This is not a fresh quality benchmark.

Both roles encode each exact body once using the same actual prepared processor.
The CPU receipt records IDs' hashes and private files; the GPU manifest binds
these descriptors, and both controller and role independently compare them.
No private token IDs/body/wire are placed in Git. CPU role/Native/weights/quality
leaves are recording; real prepared encoding is separately fed into the actual
new role loop (no D prompt replay).

The original stopped report exceeded1MiB. This entry keeps MAX_HEADER=1MiB and
references a complete immutable post-destroy JSON report. The file is fsynced,
then atomically published via exclusive hard-link; full source/size/hash/owner
identity are consumed before success. The original836 failure remains unchanged.

Budget:16 cards on910376 n12+n13, one residency; GPU5400s/Native5280s/keeper5700s
from the original epoch.22 requests/32 role stages (unless genuine early P EOS
skips D), each shape11 requests. Output cap12672 including EOS, normal-path
forward cap12760 (L:9 prefill+127 decode; S:1+1023). No automatic retry/reload.
Native/physical/context errors stop with partial evidence; failed warm gold/EOS
gracefully stops before formal. Formal wrong/cap requests remain in denominators.

Measure complete client token-ID delivery wall/queue/TTFT/ITL/decode, group wall
and useful tokens/all16 cards, transfer enclosures without adding overlapping
clocks, resident cardseconds including idle/load/cleanup. This is not HTTP/SSE,
text-token SLO, RDMA, explicit pinned-memory proof or a Model speedup claim.

Source/claim and actual CPU receipt must be published before unique fresh admission.
The keeper checks parent/node membership, labels, strict DRM inventory, KFD/PIDs,
held ports, actual8 gfx936 and existing owner affinity. Normal success requires
both8-owner destructions, srun/step0, both role files, post KFD/PID/ports bindable.
TIME_WAIT observations and passive post wait remain visible, not called clean
until bindable. No broader resource or third replica is admitted.
