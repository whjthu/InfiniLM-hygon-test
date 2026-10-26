# TP16 head-only component r9176

Reuses unchanged VocabParallelR8296 (SHA0d362b...) full-BF16 allgather/reorder/fence. New CAPI dual mode compares original ReplicatedLinear versus this head at M1/4/8/16/32/64,30 paired ABBA blocks with globalTP16 barriers outside timers. Actual lm_head weights, seed917600 synthetic hidden. Synthetic component evidence is not taskquality.

CPU full163840×7168 BF16 weight is allowed. B performs exactly one explicit H2D of10240×7168 BF16 rows from CPU rowoffset tp_rank×10240 into preallocated shard146800640B, then reads first/middle/last rows back. This component explicit-copy loader is NOT yet a proof of private Model integration; future Model must preserve globalcheckpointkey and existingWeight CPU-select_source contract or an equally explicit local loader without double-sharding. No fullGPU intermediate in shard-only mode.

Dual keeps bothweights and cannot prove saving. Separate full/shard fresh Python processes in the same isolated16DCU job produce physical driver-free snapshots constructor/load/warm/M64-output-held/destroy. Exact transientpeak and allocatorlive/reserved remain UNKNOWN. Every retained snapshot must exceed4GiB.

Host through headoutputready excludes barrier and D2H/CPUdecode; stream events and a separate equivalent GEMM/allgather/reorder pass retain launch/dependencywait, not purebusy/network wiretime. Compare max-rank costs over bothnodes, preserve ABBA paired distribution, top1differences/margins/top5/RMS. No tokenidentity gate. Portcounter traffic aggregated, not isolatedheadwire. GDR UNKNOWN.

CPU360 singleHOST TU closure passed; actual2node preflight414 pin/import/affinity64 passed, sacctauthoritativeterminal required before GPU. NoGPUsubmitted. Root must publish source and explicit fixtureclaim/GO before unique16DCU. No Model constructor or live SDK modifications.
