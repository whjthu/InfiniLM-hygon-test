# GLM layer 0 selected-value packing

Verify the base SHA-256, then apply the patch with git apply --unidiff-zero. Reuse the three headers
identified in the task record from the qualified TP16 paired component build.
The existing runtime and attention object layouts are preserved.

Both config and environment enable the candidate. Layer 0 packs its loaded
KV-B weight and scale once. Requests outside C1 decode with a selected union
of 65 through 2048 rows use the original selected-value path. Fresh KV expansion
uses the same original projection. The free-device-memory checks around packing
sample boundaries only. Logical retained packed storage is 655,520 bytes per rank.

The owner-only read function returns a three-element vector with packed storage
present, total selected packed calls and cached dispatch calls. The future typed
Model bridge must resolve the actual Runtime and use these counters to attribute
effect. The patch and hashes are local source evidence.
