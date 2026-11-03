# One original TP8 complete-attention comparison

This changed Python package binds actualCPU920's31-object library. It reuses the
original r433 checkpoint loader, actual query operands, complete per-call prefix
restore, two fixed periodic/tail U2048 selections and12 query schedule. Those
unchanged source/CPU suites are not repeated. No Model/HTTP/routing claim.

Both arms keep NoV. OFF is the real original selected-latent expansion plus
original QK/softmax/PV/O-projection/real8-rank allreduce, not a refused FullFp32
SDK operator. All8 OFF outputs must be finite, canary-safe and nonzero at real
position32767 before the first ON. ON replaces the three Casts and16 F32 GEMVs
only. Numeric .005 alert/.02 diagnostic-stop are unchanged and never task-quality
acceptance. Every completed output is saved before any STOP on every rank.

First ON verifies actual dladdr/hash for the new setter and both leaf functions,
real Native chosen/rowquant/roundtrip/8aggregation/8projection/finalcast counts,
identity selfcheck and original W scale/head mapping. The compiled successful
Owned.finish additionally requires exactly3 Cast enqueues and8+8 GEMV enqueues,
then a completed device finiteflag0. This is a HOST source execution contract,
not a hardware profiler/kernel-count claim. No new C++ instrumentation/recompile.

Existing warm-only typed snapshots retain rank0 P/latent/rowroundtrip/scales/W/V/
aggregate/precast and all8 raw actual outputs. Formal samples never arm/read a
snapshot. CAPI/checked query, coldsetup, bank allocation, prefixseed, D2H/raw IO,
cleanup, per-rank times and max8 wall remain separate; no warm speedup claim.

Resource:910374/n07,8 real TP8 participants,32CPU/64000MiB, GPU600s incl load,
worker540/shell570, keeper910407/n20 2CPU/2048MiB<=900s. Fresh membership/label/
KFD/PID/DRM8, actualCUDA8/gfx936/uniquePCI, original broad32CPU until firstsubmit,
actual8 disjoint4CPU masks and no migration. Same owner create/load/query/destroy,
Runtime weak expiry/Torch-after-destroy, original srun/step0 and independent
knownPID/KFD post. This component has no fixed HTTP/control ports; no claim that
arbitrary kernel TIME_WAIT ports are bindable. No retry/reload or extra geometry.

Only new Python policy/receipt/counter/numeric handoff consumers are checked on
CPU, including a real ctypes high64 pointer and tempfile persist-before-failure.
The actual linked SDK is not dlopened on CPU. The final HOST CLI preflight hashes
receipt/objects/source/wrapper and creates no intent; production entry remains
the actual GPU worker. Actual support/error/complete performance remain UNKNOWN
until the one claimed run. Device errors stop and preserve partial raw evidence.
