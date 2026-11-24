"""Run and reconcile two exact zero-GRES observers for the fixed-hour sampler."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import select
import selectors
import subprocess
import time

import collect_training80_busy_pulses as short
import collect_training80_busy_long_window as window
from training_dispatch_allocation import allocation


LEDGER_SHA = "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"
PCI_SHA = "df2b483f320d54383a2ccaf6e80eab3a69e6efbb3f0caf6d7292c56eb96f9b39"
SOURCE_SHA = {
    "collect_training80_busy_pulses.py": "37f1e205a7eac52914978587534f41f9cd60fa6e71650e0ddd628ae80557efaa",
    "collect_training80_busy_long_window.py": "f3e47d96a52dcf7b4596707ffea671c4d8ff7c7511ab860c242dc8bf51c25e22",
    "check_training80_point_clock.py": "0997f63efe318fb310d1e77d21021ebd4dafe753e57670952d319a343263a79f",
    "sample_training80_busy.py": "6e4a964b9fe4af69974a73dc1f5d3338d1c7cacc7d68426d0890c58cd158e324",
    "training_dispatch_allocation.py": "e603bbbb1ef6dd539d667d547675022c2a5236fba2d7bf873dbff44950ed80b6",
}
FIRST = "2026-11-25T07:32:30+08:00"
LAST = "2026-11-25T07:35:00+08:00"
HARD_END = "2026-11-25T08:36:30+08:00"
NANO = 1_000_000_000
STARTUP_NS = 30 * NANO
MIN_CLEANUP_NS = 20 * NANO
CLEANUP_LIMIT_NS = 30 * NANO
SLURM_TIME = "01:01:00"
TERMINAL = short.TERMINAL


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def time_gate(clock=time.monotonic_ns, wall=time.time_ns):
    current = clock()
    now = wall()
    first, last, end = (int(datetime.fromisoformat(t).timestamp()) * NANO for t in (FIRST, LAST, HARD_END))
    require(first <= now <= last, "controller launch is outside the authorized window")
    return current, current + last - now, current + end - now


def proc_start_ticks(pid):
    try:
        source = Path(f"/proc/{pid}/stat").read_text()
        return int(source[source.rindex(")") + 1:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class StopLatch:
    """A root-written, request-specific stop file for this controller only."""
    def __init__(self, root, identity):
        self.path = root / "stop.request.json"
        self.identity = identity
        self.reason = None

    def __call__(self):
        if self.reason or not self.path.exists():
            return self.reason is not None
        try:
            require(not self.path.is_symlink(), "unexpected stop link")
            with self.path.open("rb") as stream:
                raw = stream.read(2049)
            require(0 < len(raw) <= 2048, "invalid stop extent")
            intent = json.loads(raw)
            require(intent == {**self.identity, "action": "STOP_OBSERVER_FOR_PRIORITY_WORK"}, "stale or foreign stop identity")
            self.reason = "PRIORITY_WORK"
        except (OSError, ValueError):
            # Stop issuing new observer pulses. This does not identify a Slurm step to signal.
            self.reason = "STOP_CONTROL_INVALID"
        return True


class PipeTransport:
    """Bound all pipe reads and writes. Never retry a partially written pulse."""
    def __init__(self, children, stopping, clock=time.monotonic_ns):
        self.children, self.stopping, self.clock = children, stopping, clock
        self.selector = selectors.DefaultSelector()
        self.pending = {}
        self.buffers = {}

    def add(self, parent, child):
        os.set_blocking(child.stdin.fileno(), False)
        os.set_blocking(child.stdout.fileno(), False)
        self.selector.register(child.stdout, selectors.EVENT_READ, parent)
        self.buffers[parent] = bytearray()

    def send(self, packet, deadline):
        raw = json.dumps(packet, separators=(",", ":")).encode() + b"\n"
        require(len(raw) <= 512 and set(self.buffers) == set(window.PARENTS), "invalid observer transport")
        for child in self.children.values():
            fd = child.stdin.fileno()
            while True:
                remaining = (deadline - self.clock()) / NANO
                if remaining <= 0 or self.stopping():
                    raise TimeoutError("pulse send stopped before the next pipe write")
                try:
                    count = os.write(fd, raw)
                except BlockingIOError:
                    select.select([], [fd], [], min(0.1, remaining))
                    continue
                require(count == len(raw), "ambiguous partial observer pipe write")
                break

    def receive_until(self, deadline):
        while self.clock() < deadline and not self.stopping():
            progressed = False
            for parent, buffer in self.buffers.items():
                if b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer[:] = rest
                    require(0 < len(line) <= 4096, "observer row exceeds its bound")
                    yield parent, json.loads(line), self.pending[parent]
                    progressed = True
            if progressed:
                continue
            left = (deadline - self.clock()) / NANO
            if left <= 0:
                return
            for key, _ in self.selector.select(min(0.1, left)):
                try:
                    data = os.read(key.fd, 8192)
                except BlockingIOError:
                    continue
                require(data, "original observer pipe closed early")
                arrival = self.clock()
                parent = key.data
                # Bound arrival conservatively when a line spans two reads.
                self.pending[parent] = arrival
                self.buffers[parent].extend(data)
                require(len(self.buffers[parent]) <= 16384, "observer frame buffer exceeds its bound")

    def close(self):
        failures = []
        try:
            self.selector.close()
        except OSError as error:
            failures.append(type(error).__name__)
        for child in self.children.values():
            for pipe in (child.stdin, child.stdout):
                try:
                    pipe.close()
                except OSError as error:
                    failures.append(type(error).__name__)
        return failures


def pinned_paths(ledger, pins):
    for name, digest in SOURCE_SHA.items():
        short.require(short.pin_bytes(Path(__file__).with_name(name))[1] == digest, "pinned sampler dependency differs")
    led, led_sha = short.pin_bytes(ledger)
    pin_doc, p_sha = short.pin_json(pins)
    require(led_sha == LEDGER_SHA and p_sha == PCI_SHA, "published allocation or PCI pins differ")
    _, _, owners = allocation(led)
    short.require_pins(owners, pin_doc)
    expected = {p: sorted(n for n, o in owners.items() if o[0] == p) for p in window.PARENTS}
    require(len(owners) == 80 and all(len(nodes) == 40 for nodes in expected.values()), "wrong observer membership")
    return owners, pin_doc, expected


def command(parent, nodes, label, request, pins):
    argv = short.slurm_argv(parent, nodes, label, request, window.POINTS, pins, PCI_SHA)
    argv[argv.index("--time=00:01:00")] = "--time=" + SLURM_TIME
    argv[argv.index(str(Path(short.__file__).resolve()))] = str(Path(__file__).resolve())
    return argv


def step_row(fields):
    return dict(field.split("=", 1) for field in fields.split() if "=" in field)


def zero_tres(field):
    parts = field.split(",")
    return len(parts) == 3 and set(parts) in (
        {"cpu=40", "mem=10G", "node=40"}, {"cpu=40", "mem=10240M", "node=40"})


def qualified_cancel_step(parent, step, label, nodes, own, child, deadline, *,
                          query=short.query, account=short.account, hosts=short.hostnames, ticks=proc_start_ticks,
                          uid=lambda: os.getuid()):
    """Fail closed on a stale controller bound, absent own PID or nonzero-GRES row."""
    if not isinstance(step, str) or not re.fullmatch(parent + r"\.[0-9]+", step) or child is None or own is None:
        return False
    if child.pid != own["PID"] or own["start_ticks"] is None or ticks(child.pid) != own["start_ticks"] or child.poll() is not None:
        return False
    job = [row for line in query(["scontrol", "-o", "show", "job", parent], deadline, 65_536)
           if (row := step_row(line)).get("JobId") == parent]
    if len(job) != 1:
        return False
    [owner] = job
    if (owner.get("JobState") != "RUNNING" or not owner.get("UserId", "").endswith(f"({uid()})")
            or owner.get("NumNodes") != "40" or "gres/dcu=320" not in owner.get("AllocTRES", "").split(",")
            or hosts(owner["NodeList"], deadline) != set(nodes)):
        return False
    rows = [row for line in query(["scontrol", "-o", "show", "step", step], deadline, 65_536)
            if (row := step_row(line)).get("StepId") == step]
    if len(rows) != 1:
        return False
    [row] = rows
    if (row.get("Name") != label or row.get("State") not in ("RUNNING", "COMPLETING")
            or not zero_tres(row.get("TRES", "")) or hosts(row["NodeList"], deadline) != set(nodes)):
        return False
    actual = account(parent, label, deadline, step)
    return bool(actual and actual[:2] == [step, label] and actual[2] in ("RUNNING", "COMPLETING")
                and zero_tres(actual[6]) and hosts(actual[5], deadline) == set(nodes))


def cancel_owned(parent, step, label, nodes, own, child, deadline, *,
                 gate=qualified_cancel_step, query=short.query):
    if not gate(parent, step, label, nodes, own, child, deadline):
        return "UNVERIFIED_NO_SIGNAL"
    query(["scancel", step], deadline, 65_536)
    return "EXACT_OWN_STEP_SIGNAL_SENT"


def finish(steps, labels, expected, children, processes, hard_deadline, cancel, *,
           clock=time.monotonic_ns, account=short.account, pause=time.sleep, cancel_step=cancel_owned,
           shutdown=lambda: []):
    rows, actions = {}, {}
    deadline = min(hard_deadline, clock() + CLEANUP_LIMIT_NS)
    if cancel:
        for parent in expected:
            try:
                actions[parent] = cancel_step(parent, steps.get(parent), labels[parent], expected[parent],
                                              processes.get(parent), children.get(parent), deadline)
            except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
                actions[parent] = "SIGNAL_OR_VERIFICATION_UNKNOWN"
    pipe_close_failures = shutdown()
    while clock() < deadline:
        for parent, step in steps.items():
            try:
                current = account(parent, labels[parent], deadline, step)
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            if current:
                rows[parent] = current
        if set(rows) == set(expected) and all(row[2] in TERMINAL for row in rows.values()):
            break
        if deadline - clock() <= NANO or not steps:
            break
        pause(.5)
    return rows, actions, pipe_close_failures


def released_step(parent, step, label, nodes, deadline, *, query=short.query, hosts=short.hostnames):
    """Require an actual End and no contradictory Slurm controller state before reuse."""
    if not isinstance(step, str) or not re.fullmatch(parent + r"\.[0-9]+", step):
        return False
    columns = ",".join((*short.COLUMNS, "End"))
    rows = [row for raw in query(["sacct", "-n", "-P", "-j", step, "--format=" + columns], deadline, 65_536)
            if len(row := raw.split("|")) == 8 and row[0] == step]
    if len(rows) != 1:
        return False
    [row] = rows
    if (row[1] != label or row[2] not in TERMINAL or not zero_tres(row[6])
            or row[7] in ("", "Unknown") or hosts(row[5], deadline) != set(nodes)):
        return False
    controller_rows = [
        fields for raw in query(["scontrol", "-o", "show", "step", parent], deadline, 262_144)
        if (fields := step_row(raw)).get("StepId") == step
    ]
    if not controller_rows:
        return True
    if len(controller_rows) != 1:
        return False
    [current] = controller_rows
    return (current.get("Name") == label and current.get("State") in TERMINAL
            and zero_tres(current.get("TRES", "")) and hosts(current["NodeList"], deadline) == set(nodes))


def run(root, ledger, pins, request):
    require(os.name == "posix" and short.token(request, 32), "this exact standby observer needs a POSIX HOST and unique request")
    ledger, pins = ledger.resolve(strict=True), pins.resolve(strict=True)
    owners, pin_doc, expected = pinned_paths(ledger, pins)
    begin, last_launch, hard_deadline = time_gate()
    start_deadline = min(begin + STARTUP_NS, last_launch + STARTUP_NS)
    labels = {p: f"training80-long-busy-{request}-{p}" for p in expected}
    children, processes = {}, {}
    receiver = short.Receiver(owners, request, PCI_SHA, pin_doc)
    pid = os.getpid()
    ticks = proc_start_ticks(pid)
    require(ticks is not None and root.name == "training80-long-busy-" + request, "unbound HOST controller")
    identity = dict(request=request, controller_PID=pid, controller_start_ticks=ticks)
    root.mkdir(mode=0o700)
    window.write_receipt(root / "host-intent.json", dict(identity, parents=expected, labels=labels, GPU_requested=0,
                         hard_deadline_CST=HARD_END, hard_deadline_monotonic_ns=hard_deadline, replay_allowed=False))
    stop = StopLatch(root, identity)
    transport = PipeTransport(children, stop)
    failure = None
    try:
        short.check_live_parents(expected, start_deadline)
        require(all(short.find_accounting_step(p, label, start_deadline) is None for p, label in labels.items()),
                "original observer label already has a step")
        for parent, nodes in expected.items():
            require(not stop() and time.monotonic_ns() <= last_launch, "observer launch is late or preempted")
            argv = command(parent, nodes, labels[parent], request, pins)
            with (root / (parent + ".stderr.private")).open("xb") as err:
                child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
            children[parent] = child
            processes[parent] = dict(PID=child.pid, start_ticks=proc_start_ticks(child.pid))
            window.write_receipt(root / (parent + ".controller.json"), dict(parent=parent, **processes[parent]))
            transport.add(parent, child)
        for parent, row, arrival in transport.receive_until(start_deadline):
            receiver.accept(row, parent, "ready", receiver.ready, arrival)
            if len(receiver.ready) == 80:
                break
        require(len(receiver.ready) == 80 and not stop(), "80 bound Ready events did not arrive")
    except (OSError, TypeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        failure = type(error).__name__
    window.write_receipt(root / "ready.json", dict(identity, ready_nodes=len(receiver.ready),
                         actual_READY_steps=receiver.parent_steps, processes=processes, startup_failure=failure))
    ready_end = time.monotonic_ns()
    first = max(ready_end, begin)
    collector_end = first + window.DURATION_NS + NANO
    can_collect = failure is None and not stop() and collector_end + MIN_CLEANUP_NS <= hard_deadline
    try:
        # Even a refused startup gets the complete missing-hour denominator without sending a pulse.
        observation = window.collect_window(root / "measurement", receiver, transport, first, collector_end,
                                             stopping=lambda: not can_collect or stop())
    except (OSError, ValueError) as error:
        observation = dict(hour_interval_coverage_complete=False, writer_failure=type(error).__name__,
                           expected_samples=window.POINTS * window.CARDS, valid_samples=0,
                           missing_samples=window.POINTS * window.CARDS,
                           expected_device_seconds=window.EXPECTED_DEVICE_SECONDS, qualified_device_seconds=0,
                           unknown_device_seconds=window.EXPECTED_DEVICE_SECONDS)
    interrupted = not observation["hour_interval_coverage_complete"]
    rows, actions, pipe_close_failures = finish(dict(receiver.parent_steps), labels, expected, children, processes,
                                                hard_deadline, interrupted, shutdown=transport.close)
    released = {}
    for parent, step in receiver.parent_steps.items():
        try:
            if released_step(parent, step, labels[parent], expected[parent], hard_deadline):
                released[parent] = step
        except (OSError, KeyError, ValueError, subprocess.SubprocessError):
            continue
    try:
        verified = short.qualified_accounting(rows, receiver.parent_steps, expected,
                                              lambda value: short.hostnames(value, hard_deadline))
    except (OSError, ValueError, subprocess.SubprocessError):
        verified = False
    receipt = dict(identity, unique_request=True, GPU_requested=0, worker_steps=receiver.parent_steps,
                   actual_accounting=rows, exact_step_actions=actions, pulse_result=observation,
                   startup_failure=failure, stop_reason=stop.reason, hard_deadline_CST=HARD_END,
                   pipe_close_failures=pipe_close_failures,
                   zero_GRES_terminal_accounting_verified=verified, verified_released_steps=released,
                   full_hour_observer_C0=verified and not interrupted and set(released) == set(expected),
                   unresolved_steps=sorted(set(expected) - set(released)),
                   external_jobs_signaled=False, replay_allowed=False)
    window.write_receipt(root / "host-outcome.json", receipt)
    return receipt


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("check", "run", "worker"))
    cli.add_argument("--request", required=True)
    cli.add_argument("--pci-pins", required=True, type=Path)
    cli.add_argument("--ledger", type=Path)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--points", type=int)
    cli.add_argument("--pins-sha")
    cli.add_argument("--parent", choices=window.PARENTS)
    args = cli.parse_args()
    require(short.token(args.request, 32), "wrong request identity")
    if args.mode == "worker":
        require(args.points == window.POINTS and args.pins_sha == PCI_SHA and args.parent, "wrong worker scope")
        short.worker(args.request, args.points, args.pci_pins, args.pins_sha, args.parent)
        return 0
    require(args.ledger is not None, "exact allocation required")
    if args.mode == "check":
        _, _, expected = pinned_paths(args.ledger, args.pci_pins)
        report = dict(request=args.request, parents={p: len(n) for p, n in expected.items()}, GPU_requested=0,
                      points=window.POINTS, planned_device_seconds=window.EXPECTED_DEVICE_SECONDS, hard_deadline_CST=HARD_END,
                      commands={p: command(p, n, f"training80-long-busy-{args.request}-{p}", args.request, args.pci_pins)
                                for p, n in expected.items()}, remote_calls=0, scheduler_calls=0)
        print(json.dumps(report, sort_keys=True))
        return 0
    require(args.output_dir is not None, "exact new observer directory required")
    result = run(args.output_dir, args.ledger, args.pci_pins, args.request)
    print(json.dumps(result, sort_keys=True))
    return int(not result["full_hour_observer_C0"])


if __name__ == "__main__":
    raise SystemExit(main())
