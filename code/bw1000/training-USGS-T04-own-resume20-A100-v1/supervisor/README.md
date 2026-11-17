# T04 supervisor RSS-v2 — source only

This is a new complete supervisor copy with a narrow RSS sampling correction. The original supervisor, actual fresh-domain package and request889e2c96be9d4546bee228f6ec527bc0 remain unchanged and FAILED. Rank0 receipt remains missing. No training was replayed.

The actual failure was OWNED_VMRSS_SCHEMA, while observed peak147404886016B was below256GiB and soft_limit_exceeded=false. The old parser checked terminal state only before reading status; after-state was ignored. Exit during sampling can omit VmRSS. The old result did not save the offending process/status, so this race is not asserted as the unique established cause.

## Minimal behavior delta

Before/after identity stays tied to the registered PID/startticks and opened proc-directory FD. Foreign/reused identity never reads an unowned status or contributes a measurement. Before-terminal and during-sample terminal states classify the process as exited/unsampled, not a measured zero. A present duplicated/malformed metric still fails, including malformed units/numbers during exit. Nonterminal missing VmRSS is UNKNOWN, never zero or a claimed memory threshold breach.

Each exact owner receives at most one subsequent poll to recover a missing metric. A second missing sample (including a later new episode for that same owner) fails closed. Successful confirmation or verified exit removes the pending entry; the allowance is never reset. No inner retry loop or extra process scan is added. Missing-process identity and exit classification do not permit process killing outside the original owned cleanup.

Records retain only bounded PID/startticks/before-state/status-state/after-state, State/VmRSS row counts and SHA, whole bounded status SHA/bytes, and classification. Raw status/cmdline/environment is not written. Keep last128 diagnostics plus cumulative events and last failure; max1024 missing-owner allowances. Group RSS is null while live coverage is unknown, with a separate known partial sum and explicit unknown count. The observed peak is a sampled lower bound, not an actual RSS/cgroup hard maximum. The unchanged256GiB measured-sum soft threshold can stop even when coverage is incomplete. The original poll now persists resource diagnostics in finally before normal outer cleanup processes a parsing failure.

The original deadline/held_child/owned_processes bytes and complete main cleanup tail are unchanged. No training, optimizer, RNG, input, affinity or deadline change is included. Source gate only adds the new check_rss_CPU.py member.

## Qualification and future use

All GO fields remain false. Root must explicitly integrate this new driver and new source/claim pins into a separately approved future control or resume package. This source does not authorize a new run, and cannot reopen the failed fresh-domain training.

17 new pure-stdlib fixtures passed: normal metric, before/status/after exit including empty status, one allowed missing poll, persistent/recurrent missing rejection, identity reuse without status read, malformed/duplicate/unit/numeric rejection, unchanged256GiB soft stop. These used byte/dict fixtures only; no real /proc, Torch, child process, GPU or remote call. Original17/7 and earlier resource fixtures were not executed. check_supervisor_CPU.py is retained byte-identical provenance only; its old RSS assumptions are superseded by check_rss_CPU.py and it must not be used as RSS-v2 qualification.
