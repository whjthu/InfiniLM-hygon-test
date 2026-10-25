# GLM selective INT8 Q1: actual component result

GPU 910376.664 COMPLETED0 in57s; n12 oneDCU, owner shutdown PASS, post665 KFD empty. Default OFF; no model adoption.

The actual current651 physical checkpoint `/old/models/ZhipuAI-GLM-5.3` was bound by config/index/header SHA; layer3 expert0 is FP8 E4M3 with 128x128 scales. Upstream repo revision is not present locally and is not inferred from the directory name.

FP8 was reconstructed and requantized to channel INT8; dynamic input quantization, I8Gemm, scale/output epilogue, runtime workspace and synchronous bridge drain were timed together. All ten shapes passed finite/input-readonly/tail-canary checks; both M1 shapes passed exact CPU INT32-oracle error checks.

|M|Up INT8 us|Up BF16 reference us|Down INT8 us|Down BF16 reference us|
|---:|---:|---:|---:|---:|
|1|112.56|55.77|73.46|55.14|
|4|110.51|55.55|72.92|55.21|
|8|108.89|56.57|73.81|55.18|
|16|109.53|56.25|74.12|54.87|
|128|110.19|59.97|86.79|56.58|

BF16 reference uses Torch with resident dequantized weights. It is NOT the original GLM FP8 fused MoE kernel, so these ratios establish neither whole-model gain nor loss. Host bridge/dispatch and synchronization are included; this is not an isolated MMAC kernel measurement.

Weight requantization relative RMS: up1.133%, down0.900%. Outputs use synthetic activations, not task-quality tests. Provider path currently loses to resident BF16 reference at all tested expert row counts. A possible follow-up is grouped routed INT8 with activation quantization reused across experts, then same-interface original FP8 full-layer comparison; do not schedule another fullModel load for the naive perexpert path.
