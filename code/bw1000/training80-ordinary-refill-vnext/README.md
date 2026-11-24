# Training80 ordinary refill coordinator vNext

This candidate keeps the active 05:44 coordinator, its locks, original journal and queue unchanged. The vNext entry reuses the pinned ordinary terminal-release reader, publication checks, fresh device admission/queue-hold checks and exact compute bridge. Its separate SQLite ledger is shared by all vNext groups in this workspace.

The transaction inserts the request, every eligible candidate's `(parent, node, predecessor)` and the claim event together. Its primary key spans waves, calendars and requests. A single conflicting candidate rolls back the entire group. The claimed roster is the exact terminal-qualified candidate set. If fresh device admission subsequently selects a smaller set, the bridge stays closed for that entire claimed roster.

The original admission transport's bounded nonzero return branch emits `ADMISSION_FAILED_BEFORE_BRIDGE` with hashes of that call. A transport timeout, unexpected exception or ordinary process exit code alone cannot generate that state. The unique bridge path commits `BRIDGE_INTENT` with exact controller, wire and argv digests before calling the transport. An intent without a confirmed bridge acknowledgement remains `BRIDGE_UNKNOWN`. A validated acknowledgement only confirms the bridge control launch and carries zero independently verified GPU steps. An interrupted claim with no later event remains `CLAIMED_UNRESOLVED`. All those states preserve their claims and prevent automatic resends, including across request identifiers. No expiry or replacement path is implemented.

## Local validation

`python -B -X utf8 code/bw1000/training80-ordinary-refill-vnext/coordinator.py check` verifies and compiles the exact source anchors without invoking the frozen action.

`python -B -X utf8 code/bw1000/training80-ordinary-refill-vnext/check_local.py local-only/training80-vnext-independent-receipt.json` runs the vNext claim/dispatch entry and executes the exact frozen admission and bridge statement blocks against captured transports. It reads the original groups 4, 13 and 5 source receipts by digest, races overlapping requests, and kills a fixture process immediately after another SQLite connection observes the durable intent. The tests make no remote or scheduler calls. The historical comparison is anchored by the independent classification evidence under `results/bw1000/2026-11-25`.

## Future adoption

After the operator has established the sole coordinator for a future calendar, use `init-ledger` once to create the vNext ledger at its fixed private path. Preserve that database and its transaction sidecars. A missing or inconsistent ledger prevents later dispatch and requires independent reconciliation. Keep all vNext groups on the same local filesystem and ledger. The future invocation takes an absolute private calendar path and SHA, a new `training80-vnext-...` wave identity, an exact parent-local group number and an explicit 32-hex `--request` value. Use the same request with `status` for readback. The active historical 05:44 calendar is rejected by the future entry. This local delivery establishes no claim on a running wave.
