# Minimal long AR/spec diagnostic driver

The exact four arms are retained8635 AR, near32k AR, near32k spec, retained8635
spec. Each input is consumed in full and has output budget 128. The second input
is a constructed maintenance-archive extension of the retained real tool task.
It is not benchmark or real-traffic evidence. No answer or model output was used
to select the inputs. The official pinned tokenizer produced 8,635 and 32,638
input tokens; both fit input plus 128 within 32,768. No task suffix was cut.

Both methods use the same candidate Native and one target residency. AR keeps
feature capture OFF and uses genuine W512 prompt append and W1 greedy decode.
Spec uses publisher335 and worker334. Draft starts once after both AR arms, so
its bounded idle socket does not wait through long AR prefills. Request costs
count 32 target cards for AR, 33 service cards for spec, and record draft loading
and handshake separately. The order is a minimal one-pair diagnostic per input;
it cannot establish stable speedup or remove all order/cold-call effects.

The per-request clock includes complete prefill/capture/transport through emit,
and separately records normal reset. Selected-ID bursts are stamped when ready,
before capture/commit. These are backend timelines, not HTTP client latency.
Original deadline and cancellation are checked collectively before Native
operations. A partial request or failed answer stops the remaining arms. Natural
EOS and one actual calculate call with exact integer arguments are required by
the existing pinned XTML parser. A 128-token cap is retained as failure, not EOS.

This is an engine-level entry invoked with an already constructed Native and
capture transport. The caller still must provide the exact source/DSO binding,
parameter loader, original absolute deadline, current device mapping, one draft
start/connect action, and coordinated destruction and KFD/port cleanup. There
is no GPU launcher or execution GO in this package. `capture_transport.py` is
the explicit node2-to-node0 feature stream used by Native harvest, separate from
the 32 target collective owners and the external draft RPC.

Native327 is linked from qualified Native247 `8df87f...`, retaining its objects
except the two reviewed HOST capture units. Its new SHA is `4dfdc9...`. Current
latest99 supplement518 used `ec239c...` with additional profile exports; its
service-wrapper and profile qualification do not automatically transfer to327.
The planned same-DSO AR/spec diagnostic is scoped independently and does not
claim current latest99 whole-service or online qualification.

CPU entry tests use recording Native/four-leader control and real policy code;
the parser and complete source entry execute, but no Native Context or GPU does.
Production frozen inputs are checked separately by the actual official tokenizer.
