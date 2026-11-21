"""Run the fixed Graph32 and Graph64 forward comparison for an assigned cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
PINS = {
    "contract.py": "9c66ea595290fdcaa449c311b478352db72bdf9529b125bce0156ba03b190e22",
    "scope.py": "444ced1ffc4bf5231b7fcf7a2410e95a3cfd99370b0e015614bf4c2aa301ea07",
    "contention_worker.py": "4fe6194201848a0b065abdde41beea71400e02aac2b3c64f985b7a9470d305ad",
    "duplex_worker.py": "48310c24b6a0582a65dd8b5fa02d216784a5ae3b9dd57c570aed13b7fe6adb7f",
    "hip_runtime.py": "e551c1faa38e6472be7861a572354c72311a9b21d4ad2e73d7bb0a3fa403d534",
    "node_sample.py": "e10a17f576da010cd50937a8f91eac1c8f66098dde6d887c8373d00a615c8931",
    "qualified_sampler.py": "09fc94b7f527c159bd489bacbf7718dad8615ba61cd3f361a44f441da14a0430",
    "schedule.py": "ead627410a66ae1b278a31fb45d30f1b44b8c366c47eed74ad3fd5855fb62824",
    "setup_gate.py": "a802daa826f21a31292e81e421a57dc6bb61bd91a5a57426f5d0863239cc239d",
}
for name, pin in PINS.items():
    member = HERE / name
    if member.resolve(strict=True) != member or sha256(member.read_bytes()).hexdigest() != pin:
        raise ValueError("unbound worker source: "+name)
sys.path.insert(0, str(HERE))

from contract import DEPTH, K, M, N, NANO, PROBES, QUARTER_SECONDS, SETUP_OPEN, WORKER_SECONDS, cohort_for, counters, grid, need
import contention_worker as prior
from duplex_worker import bind, fresh_inventory, KFD, original_affinity, pci_at, save, V
from schedule import Clock
from scope import validate_worker_plan
from setup_gate import ARM_LEAD_NS, SETUP_LEAD_NS, PhaseGate


def projections(torch, device, seed):
    generator = torch.Generator().manual_seed(seed)
    cpu_weight = torch.randint(-2, 3, (K, N), generator=generator, dtype=torch.int8)
    cpu_inputs = [torch.randint(-2, 3, (M, K), generator=generator, dtype=torch.int8) for _ in range(2)]
    gold = [(value.float() @ cpu_weight[:, :16].float()).bfloat16() for value in cpu_inputs]
    weight = cpu_weight.to(device, dtype=torch.bfloat16)
    inputs = [value.to(device, dtype=torch.bfloat16) for value in cpu_inputs]
    outputs = [torch.empty((M, N), device=device, dtype=torch.bfloat16) for _ in inputs]
    references = [torch.mm(value, weight) for value in inputs]
    live = (weight, *inputs, *outputs)
    need(all(value.is_contiguous() and value.dtype == torch.bfloat16 for value in live),
         "resident contiguous BF16 required")
    addresses = tuple(value.data_ptr() for value in live)
    torch.cuda.synchronize(device)

    def run(mode):
        for index in range(DEPTH[mode]):
            lane = index % len(inputs)
            torch.mm(inputs[lane], weight, out=outputs[lane])

    def poison():
        for output in outputs:
            output.fill_(float("nan"))

    def flip():
        for value in inputs:
            value.neg_()

    def verify(negative=False):
        need(tuple(value.data_ptr() for value in live) == addresses, "resident tensor pointer changed")
        for output, expected, cpu in zip(outputs, references, gold, strict=True):
            equal = (torch.equal(output, expected.neg()) if negative else
                     torch.equal(output.view(torch.int16), expected.view(torch.int16)))
            first = torch.equal(output[:, :16].cpu(), cpu.neg() if negative else cpu)
            need(equal and first, "BF16 output or CPU first16 differs")

    return run, poison, flip, verify


def capture_pair(torch, stream, run, poison, flip, verify, resources):
    graphs, timings = {}, {}
    for mode in ('graph64','graph32'):
        poison()
        run(mode)
        stream.synchronize()
        verify()
        graph = torch.cuda.CUDAGraph()
        start = time.perf_counter_ns()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            run(mode)
        stream.synchronize()
        timings[mode] = time.perf_counter_ns() - start
        resources.callback(graph.reset)
        flip()
        try:
            poison()
            graph.replay()
            stream.synchronize()
            verify(negative=True)
        finally:
            flip()
        poison()
        graph.replay()
        stream.synchronize()
        verify()
        graphs[mode] = graph
    return graphs, timings


def measure(hip, streams, replay, verify, mode, clock, stop, start, end, row, *, burst=prior.burst):
    began = clock.toward(start, stop.is_set)
    last = began
    cutoff = end-clock.anchor
    groups, compute_host, compute_stream = 0, 0, 0.
    quarters = [0]*4
    while clock.monotonic() < cutoff and not stop.is_set():
        stats = burst(hip, streams, None, None, (None,)*4, replay, False, clock)
        last = stats["end_ns"]
        groups += 1
        quarter = min(3, (last-(start-clock.anchor))//(QUARTER_SECONDS*NANO))
        quarters[quarter] += 1
        compute_host += stats["compute_host_ns"]
        compute_stream += stats["compute_stream_ms"]
    finished = clock.monotonic()
    clock.verify()
    state = "COOPERATIVE_STOPPED" if stop.is_set() and finished < cutoff else "COMPLETE"
    row.update(state=state, passed=False, **counters(mode, groups), complete_groups_by_quarter=quarters,
               actual_start_epoch_ns=clock.wall_from_mono(began), actual_end_epoch_ns=clock.wall_from_mono(last),
               collection_end_epoch_ns=clock.wall_from_mono(finished), wall_ns=finished-began,
               compute_host_ns=compute_host, compute_stream_ms=compute_stream,
               h2d_payload_bytes=0, d2h_payload_bytes=0, captured_gemms_per_replay=DEPTH[mode],
               full_BF16_equality=False, CPU_first16=False)
    if groups > 0:
        verify()
        row.update(full_BF16_equality=True, CPU_first16=True)
    if state == "COMPLETE":
        need(groups > 0 and cutoff <= finished <= cutoff+500_000_000 and cutoff-150_000_000 <= last,
             "fixed active window incomplete or overran")
        row["passed"] = True


def monitor(path, clock, windows, addresses, done):
    devices = {p.resolve().parent.name: p for p in Path("/sys/class/drm").glob("renderD*/device/gpu_busy_percent")}
    with path.open("x", encoding="utf8") as stream:
        for index, (mode, start, _) in enumerate(windows):
            for offset in PROBES:
                if done.is_set():
                    return
                instant = start+offset*NANO
                row = dict(arm=index, mode=mode, planned_epoch_ns=instant, busy_percent=None)
                try:
                    clock.toward(instant, done.is_set, 150_000_000)
                    begin = clock.wall()
                    values = {pci: int(devices[pci].read_text()) for pci in addresses}
                    end = clock.wall()
                    need(end-begin<=150_000_000 and all(0<=value<=100 for value in values.values()),
                         "driver sample skew or value invalid")
                    row.update(observed_begin_epoch_ns=begin, observed_end_epoch_ns=end, busy_percent=values)
                except (OSError, KeyError, ValueError, InterruptedError) as error:
                    row["error"] = str(error)[:100]
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())


def main(out, plan):
    process_enter = time.time_ns()
    node = socket.gethostname().split(".", 1)[0]
    parent, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    selected = validate_worker_plan(plan, node, parent, step)
    cohort = cohort_for(plan,node)
    windows = grid(plan,cohort)
    first_arm_ns = windows[0][1]
    out = out / node
    out.mkdir(mode=0o700)
    startup = dict(request=plan["request"], node=node, step=parent+"."+step, cohort=cohort)
    save(out / "process-enter.json", dict(startup, epoch_ns=process_enter))
    remaining = plan["deadline_epoch"]-time.time()
    need(0<remaining<=WORKER_SECONDS and time.time_ns()<first_arm_ns, "fixed cohort setup expired")
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, remaining)
    stopped, collector_done = threading.Event(), threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: stopped.set())
    save(out / "stop-handler-ready.json", dict(startup, epoch_ns=time.time_ns()))
    clock = Clock()
    setup_open_ns = round((plan["campaign_epoch"]+SETUP_OPEN[cohort])*NANO)
    try:
        if clock.wall() < setup_open_ns:
            clock.toward(setup_open_ns, stopped.is_set)
    except InterruptedError:
        save(out / "cooperative-before-owners.json", dict(startup, epoch_ns=clock.wall(), stage="setup_wait"))
        return 1
    setup_enter_ns = clock.wall()
    save(out / "setup-enter.json", dict(startup, epoch_ns=setup_enter_ns))
    need(set(selected["PCI"]) == set(fresh_inventory()) and not any(KFD.iterdir()), "fresh PCI/KFD gate failed")
    import_started_ns = clock.wall()
    save(out / "torch-import-begin.json", dict(startup, epoch_ns=import_started_ns))
    import torch
    import_finished_ns = clock.wall()
    save(out / "torch-import-end.json", dict(startup, epoch_ns=import_finished_ns))
    if stopped.is_set():
        save(out / "cooperative-before-owners.json", dict(startup, epoch_ns=clock.wall(), stage="torch_import"))
        return 1

    torch.set_num_threads(1)
    need(bool(torch.version.hip) and torch.cuda.device_count() == 8, "eight visible HIP devices required")
    hip = bind()
    mapping = [pci_at(hip, index) for index in range(8)]
    need(hip.scalar("GetDeviceCount") == 8 and len(set(mapping)) == 8
         and set(mapping) == set(selected["PCI"]), "actual HIP ordinal differs")
    affinity = original_affinity(mapping)
    setup = PhaseGate(clock, windows[0][1]-SETUP_LEAD_NS, 8, stopped.is_set)
    gates = [PhaseGate(clock, start-ARM_LEAD_NS, 8, stopped.is_set) for _, start, _ in windows]
    status = dict(request=plan["request"], node=node, step=parent+"."+step, grid=windows, selected_PCI=mapping,
                  planned_cards=8, planned_arms=8*len(windows), planned_busy_card_samples=8*len(windows)*len(PROBES),
                  cohort=cohort, process_enter_epoch_ns=process_enter, setup_open_epoch_ns=setup_open_ns,
                  setup_enter_epoch_ns=setup_enter_ns, torch_import_begin_epoch_ns=import_started_ns,
                  torch_import_end_epoch_ns=import_finished_ns, hip_ready_epoch_ns=clock.wall(), torch_version=str(torch.__version__), torch_HIP=torch.version.hip)
    save(out / "intent.json", status)
    sampler = threading.Thread(target=monitor, args=(out / "busy.jsonl", clock, windows, mapping, collector_done), daemon=True)
    sampler.start()

    def owner(card):
        cpu, local, numa = affinity[card]
        row = dict(card=card, PCI=mapping[card], cpu=cpu, local_cpu=local, PCI_NUMA=numa, arms=[],
                   normal_release=False, error=None, setup_marks=dict(owner_start_epoch_ns=clock.wall()))
        marks = row["setup_marks"]
        where = "setup"
        try:
            os.sched_setaffinity(0, {cpu})
            device = torch.device("cuda", card)
            with torch.no_grad(), torch.cuda.device(device):
                hip.call("SetDevice", card)
                need(hip.scalar("GetDevice") == card, "HIP current device differs")
                row["provider_SHA256"] = prior.provider_identity()
                row["memory_before"] = hip.mem()
                seed = int.from_bytes(sha256(mapping[card].encode()).digest()[:8], "little")
                run, poison, flip, verify = projections(torch, device, seed)
                marks["projection_ready_epoch_ns"] = clock.wall()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                current = V(stream.cuda_stream)
                with hip.stream() as markers, torch.cuda.stream(stream), ExitStack() as resources:
                    try:
                        where = "capture_pair"
                        marks["pair_capture_begin_epoch_ns"] = clock.wall()
                        graphs, row["capture_ns"] = capture_pair(torch, stream, run, poison, flip, verify, resources)
                        marks["pair_capture_proved_epoch_ns"] = clock.wall()
                        save(out / f"setup-card-{card}.json",
                             dict(request=status["request"], step=status["step"], node=node, card=card,
                                  setup_marks=marks, capture_ns=row["capture_ns"]))
                        row["memory_after_capture"] = hip.mem()
                        where = "setup_gate"
                        row["setup_gate"] = asdict(setup.wait(card, where))
                        for index, (mode, start, end) in enumerate(windows):
                            where = f"{mode}_{index}_gate"
                            poison()
                            stream.synchronize()
                            ready = gates[index].wait(card, where)
                            arm = dict(index=index, mode=mode, scheduled_start_epoch_ns=start, scheduled_end_epoch_ns=end,
                                       ready_epoch_ns=ready.arrived_epoch_ns, phase_gate=asdict(ready))
                            row["arms"].append(arm)
                            where = f"{mode}_{index}_measure"
                            measure(hip, (current, markers[1], markers[2]), graphs["graph64" if mode=="common64" else mode].replay, verify, mode,
                                    clock, stopped, start, end, arm)
                            save(out / f"card-{card}-arm-{index}.json", arm)
                            if arm["state"] != "COMPLETE":
                                break
                    finally:
                        stream.synchronize()
            row["normal_release"] = True
        except Exception as error:
            for gate in (setup, *gates):
                gate.abort(card, where, type(error).__name__)
            row["error"] = dict(type=type(error).__name__, stage=where, reason=str(error)[:120])
        save(out / f"card-{card}.json", row)
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(owner, range(8)))
    collector_done.set()
    sampler.join(timeout=2)
    valid = sum(row["normal_release"] and len(row["arms"]) == len(windows)
                and all(arm.get("passed") is True for arm in row["arms"]) for row in rows)
    save(out / "terminal.json", dict(status, valid_cards=valid, cooperative_stop=stopped.is_set(),
        completed_arms=sum(sum(arm.get("passed") is True for arm in row["arms"]) for row in rows),
        missing_cards=8-valid, setup_gate=setup.summary(), arm_gates=[gate.summary() for gate in gates]))
    signal.setitimer(signal.ITIMER_REAL, 0)
    return int(valid != 8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output, json.loads(args.plan.read_bytes())))
