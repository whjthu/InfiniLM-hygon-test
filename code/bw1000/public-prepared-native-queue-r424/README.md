This static factory separates prepared GLM/Deep request admission from the old
private HTTP application. It performs no Native construction, weight loading,
device initialization, server creation, model readiness or thread start.

The application must supply the actual pinned `AutoInfinilmProcessor` instance
and the actual public `FinishReason` type. `prepare_engine` reads the actual
generation config and returns `engine`, `processor`, `tokenizer`, `parser`,
`FinishReason` and an explicit identity with `Model_READY=False`. GLM retains
the exact official template hash. The existing complete prepared encoder,
history contribution reconstruction, raw integer output budget, greedy-only
admission, queue and cancellation contracts remain in use. No history is
truncated or substituted. The processor render remains part of admission cost.

`bind_public_sampling(server, engine)` binds the original raw-budget and
pre-admission checks to that one public server instance before app creation.
It uses the original public sampling method, and does not modify its class or
module globals. This preserves rejection before SSE admission; a public route
must still prepare agent constraints using its existing upstream logic.

The old `existing_server_app` path is omitted. The public r420 server owns HTTP
and lifespan. For GLM, the assembly must bind the original r417 prequeue profile
policy to this engine, install the same profile middleware on the public app,
and pass the same profile/identity to the r423 Native loop. Native context
creation/loading and its service loop stay on their owning thread. Global
readiness may be exposed only after actual all-rank construction, complete
load/finish, memory/provider checks and service agreement. These all-rank
assembly and public provider/CLI pieces are not implemented here.

The copied queue journal retains its historical `original_reset_all32_completed`
field name. The field denotes a callback asserting a global drain; it does not
prove a world size of 32. Actual world size and per-owner signatures must come
from the Native owner result, as in the original qualified runs.

Five changed CPU composition cases exercise this factory, the real static
history/queue implementation, and the exact public sampling method. They cover
both families, full historical reasoning, ordinary generation EOS/retire,
invalid budgets/sampling before publication, stop, template and generation
metadata rejection. The processor render/token IDs and SamplingParams storage
are recording leaves. The real GLM template bytes were independently read and
hash checked, but this CPU check does not execute the official processor,
checkpoint encoding, HTTP, Native, GPU or a model. It does not repeat CPU844 or
the existing 64-request HTTP client validation.
