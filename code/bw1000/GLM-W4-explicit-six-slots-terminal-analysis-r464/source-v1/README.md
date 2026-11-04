This private immutable r464 consumer extends the accepted r460 reader only to
the exact already-started r462 slots 2/3/7. It does not launch, stop, reload,
encode or send HTTP. New terminal collection has not been executed here.

`ATTEMPT_BINDINGS.json` fixes all six allowed attempts: original slot0,
r460 slots1/6 and r462 slots2/3/7. No inferred or replacement attempt is allowed.
The new bindings come from the root's original 07:05:59 UTC snapshot, with
actual GPU/client/keeper, epoch, accepted-ready timestamp, nodes, config and
source hashes. Accepted startup is historical; total service elapsed never
reapplies the 2700-second startup gate.

After root observes a real terminal, the ordinary existing SSH runner may run
the **single standalone** `collect_terminal_once.py --slot 2` (or 3/7).
If watch is live it returns immediately before accounting or payload reads.
If an exclusive collection intent exists without the final archive, reconcile
its original `origin_HOST`, PID and `/proc` startticks; do not reseal or infer
failure from an interrupted transfer. Each final archive is created with `xb`:

`R/results/GLM464-slot<2|3|7>-terminal-original-sealed-v1.json.gz`

The collector records original GPU/client/keeper/post accounting, reports,
lifecycle records, full request bodies/headers/wires, actual client exec
identity, frozen deployment/config sources and source manifests. Private raw
payloads remain R/L. Existing slot0/1/6 seals are reused, never recollected.

The offline reader accepts the same private index schema as r460:

```text
python -B code/bw1000/GLM-W4-explicit-six-slots-terminal-analysis-r464/source-v1/analyze_original.py <private-index.json> --output <private-result.json>
```

Keep original `CPU918_archive`, existing slot0 archive/spec and add each real
new archive's exact SHA plus its returned `slot_index`. Supply only actual
available originals. Source preparation is not a terminal result.

The reader retains independent original scoring, input/body/IDs, per-layer
Native registry and all16owner identity, natural EOS/cap, trace/window/reset,
CAPI/raw token clock checks, post16PCI/KFD/PID/twoports and exact terminal
Slurm identity. New per-attempt deployment/config checks never replace these.
Missing, failed and wrong/capped/parse outcomes retain the full 256-question,
512-request denominator. With all six explicit slots, maximum coverage is
384/512; slots4/5 remain 128 missing. No full quality/NI/SLO claim follows.

The only executed regression was the real old886 archive+CPU918: complete
output remains byte-identical to parent reader SHA `9e0bec7d...`, 64 verified,
448 missing, 16 normal owners. New slots' terminal branches remain unexecuted
until actual seals exist. No synthetic terminal, new job, HTTP or test suite.
