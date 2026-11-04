# Fixed HC0 fullpre experimental public profile

This isolated package uses the ordinary `public_deep_model` InferenceServer and
the pinned a314 Native, with the qualified TP8/PP1, C1, HC0, r366=0, 32768 context
and original memory floors. The default startup configuration leaves fullpre OFF.
Explicitly select `--experimental-profile deep-hc0-fullpre-c1` (or the same
`experimental_profile` config value) to fix fullpre ON for that engine lifetime.
Use the ordinary `python -m public_deep_model --config ... --config-sha ...
--source-sha ... --run ... --actual-step ... --gpu-start-epoch ... --serve`
entry, with this package first on PYTHONPATH, followed by the pinned r427 source,
Common and offline dependencies. Only a separately authorized deployment may set
execute_authorized; this source/CPU package has GPU_GO and Model_GO false.

Requests cannot choose the profile, HC/parallel/cache geometry, or a benchmark
mode. There is no six/64-request schedule, fixture hash, nonce, 8489-input or
128-output restriction. Original public budgets default4096, accept valid1..8192,
and must fit the complete prompt plus output within32768, without truncation.
Original greedy/n=1, FIFO, timeout/disconnect intent, owner-only finish after
reset/drain, parser, natural EOS, P1 stream default and explicit P0/P1 remain.
Arbitrary tool requests still use the public Deep tool parser; a service cannot
assert an application-specific expected answer without a supplied gold oracle.

The actual owner thread resets/drains before arming, checks actual Native mode
and typed provider identity before READY, and records fixed profile in both
engine and Model readiness. Each request uses that same immutable mode, clears
stats only at original quiescence, validates all8 owners/86HC groups against the
real completed continuation (including legitimate zero-T1 early EOS), and
restores the engine-fixed mode without an extra wait after the original drain.
Normal owner shutdown drains and disarms; a poisoned reset is never retried by
the policy. HTTP/parent threads never set Native mode.

The bounded changed CPU consumer uses actual main nonserve CLI, metadata binder,
factory/provider, FIFO, worker/generate/reset, READY reader/publication and
pre-admission controls. Processor, HTTP and Native operations are explicit
recording leaves; no new codec encoding, weights, dlopen, GPU or Model runs.
This is an explicit experimental entry, not production/default promotion, NI,
cross-load stability, an all-requests-correct claim, or a15-token/s guarantee.
Old393/1374/451 failures and the450 seven discordants remain unchanged.
