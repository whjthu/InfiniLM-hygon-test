# Private Kimi loading Runtime integration

Default OFF; source/ELF/CPU handoff preparation only. No new Model residency is authorized.

`stage_private.py` verifies the frozen r428/r434 trees, candidate5b8, existing config and model manifest, then creates an exclusive private deployment. It copies the already built Runtime under its needed bare name `libinfinilm_runtime.so` without ELF edits. Native8df/Common2e145 and all loader/math/quality bytes remain unchanged except the three explicit instrumentation/provider delta sites in DERIVATION. No existing tree or installation is edited.

Both consumers have bare DT_NEEDED; Common has DT_RUNPATH, so the prepended private LD_LIBRARY_PATH takes priority. Neither Runtime declares SONAME. This is a static applicability proof, not actual loading qualification. `launch_private.py` requires explicit opt-in plus a separately frozen authorized config, sets the copy mode before exec/first copy, and preserves the qualified SDK/policy environment. It invokes the ordinary public CLI. Current staged config is deliberately execute_authorized=false.

The original strict SDK hashes remain. New post-import guard requires exactly one Runtime path (including rejecting aliases/deleted/old mappings), exact5b8 hash, and actual dladdr for Runtime create/copy/hit-counter symbols. Before/after the ORIGINAL parameter loader, eight-owner process aggregate hit counters must increase. Other tensor shapes legitimately fall back; fallback0 is not required for a full model loader. No artificial load/forward is added.

Real selected UInt8 W2 uses tp_dim1 narrow of a strong CPU tensor owner, width96 and scale6 out of global strides1536/96; Common Weight load calls external Runtime copy_from and completes sync. r201 actual MoE MD/source, current576 actual maps and current ELF data are in ELF_EVIDENCE. Complete Common extension source MD remains unknown; its exact binary and external ABI are fixed.

The one CPU card only performs readelf/nm/ldd trace and real private staging/configure/command/map-guard consumers. ldd receives exact libpython only to model interpreter-provided Py symbols, not as a production preload. Symbol-provider calls are recording leaves on CPU. No CDLL, Torch, Runtime, SDK constructors, weights or GPU execution occurs. Actual post-import maps/dladdr and chosen counters are reserved for the next independently necessary Kimi residency.

Component39% is not a model-load prediction. Coldfile IO, allocator warm state, aggregate H2D contention, full-load latency and Model quality remain UNKNOWN.
