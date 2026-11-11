# Initialize before both initial-parameter recorders

The original baseline callback d731f988 records its trainable digest during
on_train_begin. The earlier launcher appended the initialization diagnostic,
so this digest preceded restoration. New explicit --diagnostic-callback-position
before preserves the ordered diagnostic list and places it before the original
callbacks. Initialization then precedes FirstUpdates and BaselineCallback.
The initialization gate rejects an earlier BaselineCallback or duplicates.
Default after behavior and the original recipe flags remain unchanged.

Shared framework commit 0d0899dc2e0db8e16468ec80149edf398f31daa6,
parent bd24ec33a2fde6eb284dc3461822217dabe4cba5.
Seven actual CLI tests,26 repository tests, and the pure order gate/trace pass.
One new CLI fixture initially compared different dataset/output paths; corrected
fixture passes. Original model inputs/weights and real GPU were never used.
This source is not installed or target-CPU/GPU qualified. The original CPU128
result qualifies bd24 only. No full HIP first-update recorder exists yet;
provider capture and a separately claimed real model scope remain necessary.
