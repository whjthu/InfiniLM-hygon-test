This is an offline root audit, separate from the GPU package and its admission
path. It never launches jobs, requests, tokenizers, Native libraries or weights.
The original512 logical requests remain the denominator even if a shard fails.
All quality scores are recomputed from raw response wires and exact cases,
with actual owner continuation/counter/reset proof. Shard summaries are ignored.

Keep the private bundle and full output outside Git. Its `bundle.json` contains
`schema_version: 1`, the pinned `provider_source_SHA256`, and a `files` mapping.
Each entry has a confined relative `path`, raw `bytes` and `SHA256`.
Diagnostic fixtures must set `diagnostic_recording_only: true` and cannot
qualify actual Model quality or lifecycle. An audit must itself be reviewed
against actual originals after the run; synthetic checks are not that proof.

Logical names:

- `inputs/cases.json`, `inputs/requests-private.json`, `inputs/encoding.json`:
  the original CPU945 inputs, with hardcoded original hashes.
- `deployment/SOURCE_MANIFEST.json`, `deployment/SHARDS.json`, and the exact
  `deployment/configs/shardJ.json` and `deployment/plans/shardJ.json` files.
- For each observed shard, `jJ/watch.json`, `jJ/client.json`, `jJ/owner.json`,
  `jJ/ready.json`, and `jJ/NNNN.request.json` / `jJ/NNNN.wire` for every
  client-completed global logical index. Missing raw evidence is not a score.
- Normal terminal qualification also needs `jJ/closed.json`,
  `jJ/frontend.json`, `jJ/STOP.json` and original `accounting.json` rows for
  GPU, client and keeper. Each accounting row uses columns
  `[JobIDRaw, JobName, State, ExitCode, ...]`.

Run using the fixed project Python:

```text
global_original.py --bundle LOCAL/bundle.json --deployment-sha EXACT_FINAL_SHA --output LOCAL/root-recomputed.json
check_contract.py --workspace WORKSPACE --work-name NEW_PRIVATE_DIRECTORY
```

The contract check reuses archived CPU945 request/SSE bytes plus explicitly
recording owner leaves. It checks global missing denominators, raw-wire
regrading, forged-score/hash/mode/counter rejection and path confinement.
It does not repeat HTTP, load a model or exercise the actual terminal branch.
No noninferiority margin or hardware/client SLO qualification is invented.
