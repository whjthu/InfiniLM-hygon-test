# Original failed domain run reader

Pure standard-library, offline only. Pins both independently fetched original JSON bundles and consumes all eight rank owners/progress, seven final receipts, final input/forward ledgers and checkpoint manifests. No training, model import, remote call or tensor load.

The original supervisor failure and missing rank0 receipt remain FAIL. Counts confirm actual 800 microbatches and 25 updates per rank; neither counts nor checkpoint manifests qualify restoration, saved/live optimizer equality, performance, held-out effect or a domain model.

Run read_original.py with --terminal, --ledger and a new --output file. Private metadata bundles remain local-only; public output contains only bounded counts and hashes. The full native input/field identity comparison is separately recorded in the domain audit.
