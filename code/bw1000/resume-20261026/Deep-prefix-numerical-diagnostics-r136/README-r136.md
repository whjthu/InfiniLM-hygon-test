# Deep prefix numerical diagnostics

Actual step1134 passed the zero-weight8 Runtime-device contract, loaded67569 weight keys, then stopped at the first continuation-logits NRMSE0.0453034. The old result has no traceback and no persisted cases. Execution follows the first p2 accepted1/2 case: reference ordinary forward has packedT3, transactional verification has T4. Existing evidence does not distinguish shape-dependent BF16/INT8 arithmetic from a prefix-state numerical bug.

This is an entry-only diagnostic revision using unchanged actual CPU462 Native f100e038. No new native build or sharedDSO mutation. Owner/length/validity/stale-ticket/cohort-fault/reset/finite/memory/lifecycle checks remain hard. NRMSE, cosine, top8 overlap, actual argmax IDs, and softmaxJS are recorded per row; no arbitrary tolerance or exact-token gate is used for task quality.

All16 original cases retain per-phase evidence before numerical comparison. Recoverable hard failures are recorded and followed by verified whole reset; unrecoverable state causes explicitly skipped remaining case records. Added controls: ordinary identical-shape repeat and transactional full-accept2/2 at positions2/3/127, plus rejected-future-token perturbation with identical packedT4. These diagnose, rather than presume, causality or rounding causes. No raw logits/activations leave the process.

Even a complete hard-state run remains numerical_quality UNRESOLVED_REVIEW_CONTROLS and model_quality UNQUALIFIED. The controller does not promote qualified=true. Natural QA and a real benchmark without meaningful loss are required before model-quality promotion. No MTPhead is loaded and no speculative throughput is claimed.

Root: stage and publish the8source/result files plus manifest, use exact actual CPU462/native/source pins in false GO template with pushed commit, then launch_CPU_keeper_r136.py once on374/n05. It performs the device contract before one weight load and all20 diagnostic cases. No GPU is authorized by this package itself.
