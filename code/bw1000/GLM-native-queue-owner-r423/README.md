Static Native queue owner extracted from r417. The public InferenceServer owns
HTTP. This module does not import/start a web server, create a thread, join a
thread, construct a frontend, run a fixture plan, or execute source strings.
It does not implement all-rank launch or claim Model_READY.

Two boundaries are implemented:

- `serve_requests(...)` borrows a loaded same-thread ctx and performs the actual
  original Native service protocol. Paths, identity, queue, prepared identity,
  EOS, profile, absolute monotonic deadline and stop intent are explicit inputs.
- `run_resident_owner(r422_context, ...)` is the intended production owner
  composition: real r422 construct/load/yield, original Native environment
  policy at the same after-load point, service loop, then r422 reset/destroy/
  shutdown. The normal shutdown flag is persisted only after the original
  context lifecycle record confirms successful cleanup. The composition does
  not return a borrowed ctx after the context scope expires.

`serve_requests` is useful as the borrowed-loop boundary for CPU checks. Real
deployment should enter it through the composition, so original Native getenv
policy and context lifetime are retained. Environment configuration requires
the actual process `os.environ`, never an unrelated dictionary. No environment
lookup or `/refactor` path exists inside the loop.

The emitter is exactly stage1/rank0, with the already prepared queue. Other
owners receive `engine=None`. The prepared identity must retain actual latest99
upstream identity and EOS set. The queue must first be bound by the original
`service_policy.bind_prequeue(engine, default_profile)`. The same explicit
default profile goes to public ProfileHeaderMiddleware and all-rank agreement.
The r424 prepared engine/FinishReason/identity outputs satisfy this boundary;
its parser/public sampling remain frontend responsibilities.

Retained numerical/state protocol:

- original all-rank integer wire and exact W2/W4 profile tail;
- complete prompt, C1 capacity32768, <=512-token forwards, decode W1;
- original ordered prefill groups, votes only at original group boundaries,
  final-chunk logits retention, original sample/broadcast and visible-EOS rule;
- original indexer/NoV/transient/batch/lease policy calls and Native flags;
- original same-stream drain, finite-logits check and per-call512MiB service
  memory floor, cancellation reset/barriers and normal STOP drain;
- Native failure is not cooperative cancellation: no subsequent request and no
  failing-rank shared WORLD drain/reset. Exceptions propagate to the real
  owner's cleanup and the caller's whole-job fail-stop. No retry/reload occurs.

Original constructor/preload hardware, memory admission, all-rank agreement,
4GiB postload floor, affinity/provider/head inventory and process-level cleanup
must still be retained by the future launch/provider orchestrator. Neither
local construction nor this service loop substitutes those existing gates.
There is no complete public family factory or readiness protocol yet. In
particular, the caller must propagate owner failure to all peers/frontend;
an uncaught Python worker-thread error alone is not a complete job fail-stop.

Diagnostics preserve attempted requests vs completed requests. Active-call
return code/finite status is retained before failure so a completed failing
call is not erased. No raw logits or NaN JSON is written. The successful CAPI
timing enclosure still contains the original CAPI+return-code check (and small
new diagnostic scalar updates); no speed claim is made. Report-write failure
cannot replace an existing Native primary error. The old queue journal's field
name `original_reset_all32_completed` is preserved byte-for-byte for compatibility;
it is not new proof of32 owners in this16-owner protocol.

Local changed CPU checks run the actual new loop with the real old queue,
NativeRequest, typed wire, profile rules and execute_plan. Torch/device storage,
Native/collective peers and tokenizer/history are recording leaves. One
composition case records only the constructor leaf, without repeating r422's
16 checks or pretending device destruction. No HTTP import, Native dlopen,
checkpoint payload, GPU, compile, remote CPU job or Model occurs.

Unmodified helper copies are pinned in AUDIT.json. Old prepared-family/chat/
history modules are only fixtures for the old queue CPU consumer; production
accepts r424's prepared queue object and does not call the old factory or HTTP
helper. `r417-to-static-loop.diff` records the exact source delta for review;
the frozen r417/r422 source and shared installation are unmodified.
