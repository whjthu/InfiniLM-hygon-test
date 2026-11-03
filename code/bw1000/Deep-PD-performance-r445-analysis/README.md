Read-only r445 original evidence reader
=====================================

This stdlib consumer adapts the verified data boundaries of
`tools/audit_pd382.py`; that file and all r382 evidence remain unchanged.
It performs no subprocess, remote read, Model import or dynamic loading.

Input is an immutable local seal using the existing r382 representation:
`run`, `all_frozen_source_members_unchanged`, `accounting`, and `files`.
Each file entry has the original `path`, `bytes`, `SHA256`, `gzip_base64`.
File basenames must be unique. The seal must include original watch,
client and both final role reports. Include the post's existing accounting
row; accounting columns are JobIDRaw, JobName, State, ExitCode, Elapsed,
NodeList. No separate post or observation is launched by this consumer.

Original names remain:

* `Deep-PD-r445-RUN-watch.json`
* `Deep382-RUN-client.json`
* `Deep382-RUN-P.json` and `Deep382-RUN-D.json`

Supply exact externally consumed publication/source and plan hashes:

```text
python -B analyze_original.py SEALED.json --run RUN \
  --source-sha FINAL_GPU_MANIFEST_SHA256 --gpu 910376.STEP \
  --keeper 910407.STEP --plan FROZEN_BENCHMARK_PLAN.json \
  --plan-sha FROZEN_PLAN_SHA256 --output LOCAL_RECOMPUTED.json
```

The reader validates the full 22-request/32-stage plan, each dynamic
prompt/budget/fixture identity, HC0/provider evidence, actual no-prefill-on-D
continuation, all 2184 state-block digests and terminal owner/post boundaries.
It independently parses exact typed tool nonce/integer gold or the original
last `#### A` MMLU answer. A failed/capped/pending request is retained; an audit
can succeed while quality, lifecycle or performance does not qualify.

Rates and queue/TTFT/ITL use only one controller clock. Cohort throughput ends
at all four requests' terminal boundary and divides by all 16 resident cards.
Pairs (co4,PD5) and (co7,PD6) preserve fixture order. The co4→co7 and PD5→PD6
endpoint changes also change fixture order; they are **not** pure repeated-load
drift. Native local phase timers and overlapping P/D transfer timers remain
separate, and private token-ID clocks do not establish public SSE token SLOs.

`python -B check_offline.py` checks only synthetic parser/timing rejection
boundaries. It does not validate actual r445 execution or performance.
