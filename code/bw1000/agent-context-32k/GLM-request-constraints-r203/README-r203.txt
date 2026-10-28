Candidate only. Do not edit or attach this to running GLM192b/801.

api_request_constraints_r203.py reuses Deep196 request_constraints exactly.
It appends generic request rules to a copied first system message, or inserts
a copied system message if none exists. It never filters GLM tools, alters
historical reasoning/tool messages, changes template/Native/budgets, strips a
second generated call, or relaxes the strict parser. auto+parallel=true has
no extra rule and must render byte-identically. Existing GLM named choice
keeps all tools visible; only its parser schema remains named as before.

integration.diff is a two-line import/encode_chat constructor hook for a
future private namespace selected by root. This bundle launches no service.

CPU_contract_r203.py --source PATH runs a fake-tokenizer CPU contract through
the pinned existing GLM queue/bridge. Local author execution: PASS5; it is
not actual tokenizer/Model qualification.

Root actual saved-request comparison: use existing outer/container on CPU,
2CPU/4096MiB/gresnone/5min, set GLM_SOURCE_RUN_R203 to the exact completed
regular-run suffix string, then CPU_inside_r203.sh. The source directory
defaults to /refactor/incoming/GLM-regular-service-r192, manifest6f1dc599...
The comparison reads exactly HTTP/SSE/thinking request002 and their wires,
checks the wire SHA pins from root's actual metadata, and renders old/new
with the unchanged official GLM tokenizer and template3740abce... . It also
checks named/full-tools, required/none, complete copied history/thinking,
raw4K/8K, context32768 and auto+paralleltrue. No prompt/text/arguments/nonce
are printed; the summary is booleans/counts/SHA only.

Torch/infinilm/_infinilm imports and model-weight file opening are blocked.
An installed tokenizer dependency attempting a blocked import stops the
CPU step with a fixed error class/hash. There is no tokenizer substitute,
Model generation, live patch, retry, Slurm/SSH/Git operation or GPU entry.
Actual saved-request CPU and new GPU effect are NOT_RUN by source author.
