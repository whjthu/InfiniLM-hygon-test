This is a private, default OFF source candidate for the two C1 replicated BF16
projections in the current Kimi MoE. Each TP16 owner computes its contiguous
output-row slice with the same Dense/GEMM interface and reconstructs the full
vector with the existing AllGather. It introduces no GPU kernel or precision
policy. M>1 and unsupported geometry keep the original layer forward.

The source retains strong input/weight owners and drains before temporary
leases release on failure. Existing full parameters remain resident: this
first component version claims **no GPU memory-capacity saving**. It adds two
collectives per MoE, so lower logical matrix work can still lose in full wall.

The CPU card compiles the actual project Dense/AllGather overloads without
linking or loading SDK code. A separate CPU F32 reference evaluates synthetic
BF16 weights at both full shapes, reconstructs rank order, and rejects a wrong
rank order. Neither proves device rounding, communication, replicated inputs,
full MoE correctness, performance, or Model qualification.

Any later device consumer must use actual TP16 RankInfo, verify identical
replicated projection inputs and weight identity across ranks, count both
collectives and time the complete MoE including communication and allocation.
The current 437 consumer affinity correction must be reused rather than a new
handwritten pin policy. No device card is claimed or started by this source.
