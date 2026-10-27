DeepSeek V4 Flash agent interface P0

Assigned allocation 910374, a02r4n01, 8 DCU, TP8/PP1. The original production
ac937 Native, 128 cache rows, capacity1024, loader, forward and final destroy
remain unchanged. The official DeepSeek DSML encoder and the nine 7ceb protocol
Python files form a private overlay on Common/Deep; no model source is replaced.

Stage CPU source manifest54c9c80d plus its20 files to incoming/Deep-agent-P0-r143d.
Run launch_CPU_r143.py once. It is pure static AST/source/object/DSO/ldd-r;
it does not import Torch, Native, tokenizer or server on hidden GPUs.

Stage this controller manifest plus its files to incoming/Deep-agent-P0-control-r143d.
After successful actual CPU receipt and Git publication, root creates a separate
root-GO-after-CPU.json using the false template and actual SHA/commit values.
Invoke invoke_once_r143.py once; follow the original owned GPU handle thereafter.

The real eight-GPU job first imports the existing Deep runtime and shared server,
checks Native CAPI dladdr and all five required DSO map paths/SHA, before opening
checkpoint payloads. It then loads the43-layer model once and retains it while
the existing7ceb HTTP/SSE routes drive real Native tokens. Prompt history plus
output budget must fit1024; oversize is rejected before inference, never truncated.

Client witness uses actual generated calls and real sandbox
read_file(input.json) -> calculate -> write_file(result.json) -> read_file(result.json).
HTTP/SSE named loops, auto/required, none final, schema/history rejection,
truncation/recovery and a conditional reversed multi-call result check are saved.
The thinking case requests256 tokens; a capped or non-following model is retained
as failure, not substituted. Text is buffered for P0; no incremental latency claim.

Current status: CPU source contracts only. Real static CPU, native import, model,
HTTP/SSE and tool P0 are NOT_RUN until root executes them. GPU COMPLETED0 and
normal Native shutdown are distinct from P0 all-cases PASS. Raw requests/responses
and per-request token IDs/timestamps stay in the experiment result directory.
