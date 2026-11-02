# Kimi TP8/PP4 ordinary C1 candidate

This private candidate tests a topology package: reduce TP from 16 to 8 and increase PP from 2 to 4 while retaining 32 target cards and the current 32K ordinary workload. Local KDA/MLA heads double from 6 to 12 and MoE intermediate width doubles from 192 to 384. Any later wall change must be reported as a topology-package result, not a communication-only result.

The new explicit factory constructs the original InferEngine directly. It does not construct paired verification, capture, draft, commit or replay state. The old TP16/PP2 long factory is preserved. Only three HOST translation units change: opaque factory/caller, cache allocation admission, and exact persistent UInt8 MoE scale allocation admission. SDK, GPU kernels, public Model class layout, model math and checkpoint descriptors are unchanged.

The cache keeps two physical KV rows, three KDA state rows, 260 pages of 256 tokens, and a 32768 context. A shared geometry helper restricts the new typed entry to one row: prompt chunks up to 512 and decode width one, preserving absolute positions and the original InferEngine length ledger. Existing legacy forward and resize/page changes reject these new contexts. No batch/capture qualification is inherited.

The TP8/PP4 compact flag initially selected a legacy 3/4-page validation before the long validator. The private allocator now reads long scope first and calls the existing three-argument cache validator; the original short path retains the same semantics.

The local C++ contract covers constructor admission/lifetime, old TP16 geometry, 68 legal prompt/decode intervals, invalid intervals/rows/modes, and 3584 I384 scale alias bounds. This is CPU geometry evidence only. The isolated CPU job compiles three real HOST TUs, checks fresh dependency sets and unchanged header bytes, reuses 30 current objects, links the full Native, preserves old exports and checks exact dynamic providers. It never initializes a device or loads model parameters.

Actual 12-head numeric behavior, physical memory peaks, final Python service integration, loading and performance remain unverified. A GPU/model run requires a separate concrete claim and complete final consumer first. The original 4GiB memory floor is retained for any future model run.
