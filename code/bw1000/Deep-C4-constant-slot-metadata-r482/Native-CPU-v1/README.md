# Deep constant-slot metadata Native CPU build

One4core/8GiB/900s CPU-only scope,870s inside/860s work.19 fresh compilation units:5 class-layout consumers plus14 objects lacking original complete dependency records (7HOST,12GPU source TUs);31 old objects require exact live SHA,50 total in the new library. No old contract replay, Native dlopen, explicit SDK/Runtime calls, Torch import, weights or GPU job.

The original1112 all50-M closure is reused as input. All19 new fresh-M must close before the first compile. A private per-file mirror preserves quote origins while replacing all3 original C4 class definitions consistently. It is built only from individually pinned original files. No directory whitelist or implicit missing-MD reuse is accepted.

Source-v2 fixes the fallback host-ID vector lifetime: it is declared before Work so normal status sync and the Work exception drain occur before destruction. Source-v1 remains an unexecuted proposal. Constructor metadata host vector also outlives its original catch drain. No additional sync, kernel math, cache guard, or SDK change.

The documented successful parent ENV recipe is reused; old ambient environment was not fully recorded. Parent/new ldd-r stdout and stderr, exact resolved path/hash sets, retained strong exports, unchanged NEEDED and new C ABI are required. Metadata subprocess implicit initialization is UNKNOWN. This build grants no device/numeric/Model/performance qualification.
