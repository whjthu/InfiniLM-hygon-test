# One-slot startup diagnostic, source/CPU stage

The eight original replicas are terminal. Root verified their 128 normal owner
closures, zero Native service requests, eight GPU COMPLETED0, eight intentionally
killed keeper controllers and 16 clean nodes. Every original MODEL_READY
appeared only after real STOP. All original512 quality requests remain missing.
This is an infrastructure failure, not quality or Model PASS.

This private source copies provider1183/deploymentb56. Four provider files add
small per-thread markers; a new stdlib helper arms faulthandler's C watchdog
only in the stage1/rank0 emitter, after publishing its real owner-ready record.
It repeats every20s so a legitimate slower owner load does not consume the only
stack observation. Successful real MODEL_READY, startup error, emitter return
or provider close cancels the watchdog. A short private lock prevents resolve
before arm from causing a late timer. No new production Python monitoring
thread, Native call, Tensor, sync, thread policy or math change is introduced.
Faulthandler captures Python frames without locals, not Native stacks or proof
of GIL causality. Existing first wire evaluation order is retained; its decoded
opcode is recorded rather than assuming the control was WAIT.

The existing external keeper checks startup before accounting queries and again
before client launch. The absolute startup cap is original GPUepoch+2700s.
The first observation of all16 real ready files invokes original Run.all_ready,
including exact run/source/config identity, loaded/postload/service agreement
and each stage's32 disjoint four-core owner masks. From that first complete
observation MODEL_READY must appear within120s. Reobserving ready does not reset
that timestamp. The existing loop's15s observation cadence and bounded source
reads are stated explicitly; neither deadline is a new GPUepoch or service cap.

Failure latches client_launch_blocked, snapshots original small marker/stack
files with SHA, then sends real identity-bound Run STOP. It gives the original
idle owners60s to drain normally. Only if terminal0 cannot be established does
the existing exact owned-step/PGID TERM→KILL path run, followed by the existing
two-node independent PID/KFD/port/PCI post. A MODEL_READY written after STOP
cannot clear the failure latch or launch a client. Unknown accounting is not
normal shutdown. Trace files larger than8MiB trigger this same early failure;
the original file remains on R, with at most one extra C dump possible before
the keeper observes the size. EOF snapshots explicitly allow partial last
lines, read at most16MiB/file and32MiB aggregate, and do not read model reports.

Only fixed slot0 may be used by this version; verify_gpu also requires exactly
one admitted slot. GPU_GO/Model_GO remain false and admitted_slots is empty.
The future device scope is the original slot0 parent910376/n14,n15,16 cards,
TP8PP2/C1/32768/Native952/NoV, exact frozen32questions/64paired requests.
If READY succeeds, all64 requests continue; startup timeout is no longer a
service limit. The original Native28800/service28200/GPU29400/keeper32400
deadlines retain the same immutable epoch. Global256questions/512requests are
not reduced or redistributed. The first actual same-question W2/W4 pair will
be reviewed before considering the other seven replicas; there is no automatic
eight-replica reload.

This claim authorizes one new CPU4core/8GiB/300s job only (inside270s), no
GPU/Native/SDK execution or dlopen, weights, compilation or old suite replay.
It runs the actual controller __main__ pure preflight and whole launcher main
through its pre-Popen return; only external scheduler read replies in the latter
are explicitly recorded. Source/manifest/config, actual client parser and frozen
fixture metadata are consumed unchanged. Actual Actors/provider thread starts,
real Run ready publication and marker/cancel callbacks run with actor Process,
loaded owner body/affinity metadata and signal leaves explicitly recorded.
The test retains the real 20s repeating C watchdog stack file and cancellation
evidence. Pure Run/deadline/latch, STOP60, existing failstop and startup-primary
error tests cover only changed glue. CPU918 encodings,922 quality main,926
32-core topology remain reused; none are rerun. CPU fake owner records never
constitute Native/GIL/affinity/device qualification. Future HOST preflight,
fresh physical membership/KFD/PID/ports/PCI and concrete singleton GPU claim
remain required after the actual new CPU receipt, with no automatic device GO.
