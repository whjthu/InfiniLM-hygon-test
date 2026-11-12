# Original Qwen native-template CPU qualification

Run the unchanged shared template-input collector on all 64 original COCO rows, with the exact qwen3_vl template, maximum length 4096, maximum pixels 262144, original BF16/SDPA settings and seeds. The nine original processor files and all declared original input files must match their hashes. Raw sample contents, token IDs, labels and pixel tensors remain transient in the original environment; only complete type/value fingerprints and lengths are saved. Five existing shared evidence checks run once in that environment.

This is pre-model template encoding. The source Qwen template inherits a final collator that needs the actual model rope-index function; this scope does not substitute a fake model, call that collator or claim actual distributed first batches. Per-row full-FP32-logit sizes are arithmetic based on the actual encoded lengths. Full serialized first-two-update capture budget remains pending.

The single foreground CPU scope uses four affinity cores, two Torch threads and an 8GiB aggregate RSS limit. One epoch starts before installation: whole 300s, CPU deadline 260s, supervisor work stop 270s and bounded owned-group cleanup. Devices are hidden and device initialization/weight opening are forbidden. No GPU, model load, training update, numerical, recovery or performance qualification is implied.
