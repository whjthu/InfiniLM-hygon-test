# Kimi bounded pack reuse: two changed translation units

Immutable CPU-only package. Compile the published pack-reuse HOST and GPU sources once, using the exact original1093 HOST and1075 GPU prefixes from the published tiling plan. Reuse34 production objects and the1076 probe HOST object by exact SHA; no old contract/device suite.

Both fresh-M closures must finish without unmatched bytes before any compile. The source directory supplies the new declaration header; all original include origins follow unchanged. Verify all MD sets/bytes after compile. Link all objects before libraries, retain all parent strong exports, add the lengths initializer export, and register exactly the original absolute Native NEEDED path replacement in the probe. Full qualified ENV/preload and transitive-library path/SHA closure are retained; child preload initialization is UNKNOWN. No Native dlopen, SDK operator, Torch, Runtime, checkpoint, or GPU run.

The claim is four CPU cores/8GiB, one300s Slurm step with270s inside/265s work caps on n09/parent910380. New objects are compile units, not devices. No retries in this namespace: retain the original intent/request/receipt on failure or UNKNOWN. A separate published component scope and fresh/dedup checks are required before device testing.
