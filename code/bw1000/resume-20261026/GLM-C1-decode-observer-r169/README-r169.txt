R169 CPU build only. Parent ce830, real 767/784 lineage.
Recompile only four existing HOST observer TUs; all four .cpp byte-exact original r131.
Header-only selector observes legal T1 decode calls 17,18,19; no prefix/T15/T16 samples.
512 preallocated events/owner; original stream, original drain, no new synchronization.
Device interval buckets include host enqueue gaps and dependency waits, not GPU busy.
HOST spans overlap device spans; never add them. Internal attention/FFN TP cost UNKNOWN.
Build exact CPU parent/source/object/dependency/MD pins, no Model creation or dlopen.
Run CPU_inside_r169.sh under original GLM container and ordinary CPU allocation.
Model 4 QA C1 OFF/ON/OFF runtime is separate; this source does not launch GPU or mutate ce830.
