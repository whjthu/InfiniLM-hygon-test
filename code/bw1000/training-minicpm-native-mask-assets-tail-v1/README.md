# Verified compressed-archive suffix continuation

Original360s receiver actually failed with TimeoutError/SSHrc1, leaving126877696 compressed bytes and0 extracted original payloads. The local original398106278-byte archive and remote preserved prefix have the same prefixSHA. Original failure/intent/partial are retained.

This scope appends only the271228582 missing compressed suffix bytes from the already downloaded, full-SHA-verified local archive. It never sends the prefix again, regenerates masks, recompresses source data, changes the original deadline or replays the original receiver. It rechecks prefix size/SHA and inode before append, then verifies the whole compressedSHA before using the unchanged pinned76-file extraction parser. Original failed receipt is preserved alongside distinct resume intent/completion/failure.

Budget reflects observed ~0.34MB/s upload: new suffix receiver2400s, local monotonic total2700s; no service, poller, GPU, Slurm, model or Torch execution. Target independent CPU reader checks all76 raw files and8 complete rank manifests after completion. New append code has9 CPU wrong-prefix/range/missing/extra/hash rejections. Exact original masks and initial trainable state remain private and do not prove numerical or Model qualification.
