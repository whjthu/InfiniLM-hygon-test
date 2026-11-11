# Native full-state resume witness CPU qualification

The shared framework now observes adapter, optimizer, scheduler and per-rank RNG restoration at native Trainer boundaries. It preserves the original training recipe and max_steps=25, replaces fresh-only observers, hashes full resumed inputs and saves complete failure denominators. Model forward and optimizer implementations are unchanged.

This scope runs25 CPU tests and loads the existing checkpoint20 into CPU metadata only. Original tensor values remain on the original server. No GPU or real-model restore is authorized by this CPU claim. The first A100 CPU package passed9 witness and9 recipe tests but failed to import the cursor suite because its CLI source was omitted. This corrected package includes all12 required framework sources and retains that failure.

An independent actual terminal reader is required before using target CPU qualification. Any later GPU scope needs its own immutable package, deadline, resource admission and claim. Continuous-tail input equality, complete final state, numerical tolerances, formal performance and heldout effects remain unqualified.
