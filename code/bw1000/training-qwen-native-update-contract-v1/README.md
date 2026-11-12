# Qwen native update contract

The six framework deltas extend the shared native observer to Qwen3-VL while retaining the original Gemma spec exactly. Qwen keeps SDPA, 4096 maximum length, gradient checkpointing enabled and the complete model.visual frozen boundary. The entry also bounds native observer output paths.

The original A100 config/index and three installed source files were read and SHA-verified. The four shard sizes are stat metadata, not full-file hashes or proof of loaded weights. No Torch import, weight-value read or GPU call occurred.

Local checks are standard-library recipe/structural doubles and source policy checks. A real full Qwen meta-model with actual Swift/PEFT must still establish the complete trainable manifest before a GPU claim. Existing first-update recorder limits remain unchanged. At 4096 positions full FP32 logits alone would be 2489319424 bytes, above its per-artifact limit; actual initial batch lengths/dtypes and total capture size remain unknown. Do not truncate inputs/logits or launch before resolving that budget.

The initial sandboxed read returned transport 255. Its side-effect-free exact metadata request was observed successfully through the authorized ordinary SSH configuration; no training request was submitted or replayed.
