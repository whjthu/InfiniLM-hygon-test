# Original Qwen full model identity

Use the shared framework's bounded indexed-safetensors inspector to make exactly one complete streaming SHA256 read of the four declared original shards. Validate every header, contiguous byte range, shape/dtype size, complete 750-key index correspondence and total payload bytes. Only names, shapes, offsets, hashes, sizes and resource/terminal metadata are saved or returned; no tensor values are decoded, exported or loaded into a model.

The original config/index identities and four shard sizes/mtime values must match the previously observed original metadata before streaming. All file identities must remain stable across their full reads. Six shared malformed-input/deadline checks run once in the source environment, without Torch. The full-read SHA identifies the currently observed original files; it does not independently prove official release provenance or historical performance-run byte identities.

One CPU scope, two affinity cores, 1GiB group RSS, a single epoch before installation: whole 300s, child deadline 260s, supervisor work stop 270s, TERM/KILL 5s each. The original namespace cannot be replayed. Partial progress retains the four-shard denominator. No GPU or native model launch is permitted.
