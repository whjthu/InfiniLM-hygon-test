"""Read one bounded fleet-wide pulse sample of the existing GPUs' busy counters."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import socket
import subprocess
import sys
import time
from uuid import uuid4

from check_training80_point_clock import HOST, MAX_LOCAL_NS, PARENTS
from sample_training80_busy import PCI, node_map, require, require_pins
from training_dispatch_allocation import allocation


PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
INTERVAL_NS = 2_000_000_000
REPLY_WAIT_NS = 3_000_000_000


def token(value, length):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def pin_bytes(path, maximum=262144):
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    require(len(raw) <= maximum, "metadata exceeds exact scope")
    return raw, hashlib.sha256(raw).hexdigest()


def pin_json(path, maximum=262144):
    raw, digest = pin_bytes(path, maximum)
    return json.loads(raw), digest


def store(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_counters(drm, expected):
    files = {}
    for render in sorted(drm.glob("renderD*")):
        device = render / "device"
        pci = device.resolve(strict=True).name.lower()
        require(PCI.fullmatch(pci) and pci not in files, "unexpected or duplicate DRM PCI address")
        files[pci] = device / "gpu_busy_percent"
    require(set(files) == set(expected), "current DRM PCI set differs from the eight frozen cards")
    busy, failed = {}, []
    for pci, path in files.items():
        try:
            with path.open("rb") as stream:
                raw = stream.read(16)
            require(re.fullmatch(rb"[0-9]{1,3}\n?", raw) and int(raw) <= 100, "invalid busy counter")
            busy[pci] = int(raw)
        except (OSError, ValueError):
            failed.append(pci)
    return busy, sorted(failed)


def run_worker(identity, expected, points, incoming, send, drm):
    require(len(expected) == len(set(expected)) == 8 and all(PCI.fullmatch(pci) for pci in expected),
            "invalid worker PCI pins")
    send(dict(identity, event="ready"))
    seen = set()
    for sequence in range(points):
        raw = incoming.readline(512)
        require(raw.endswith(b"\n"), "incomplete pulse input")
        packet = json.loads(raw)
        nonce = packet.get("nonce")
        require(set(packet) == {"request", "sequence", "nonce"} and packet["request"] == identity["request"]
                and type(packet["sequence"]) is int and packet["sequence"] == sequence
                and token(nonce, 32) and nonce not in seen, "unexpected or duplicate pulse")
        seen.add(nonce)
        opened_ns = time.monotonic_ns()
        error = None
        try:
            busy, failed = read_counters(drm, expected)
        except (OSError, ValueError):
            busy, failed = {}, sorted(expected)
            error = "DRM_INVENTORY_UNVERIFIED"
        closed_ns = time.monotonic_ns()
        send(dict(identity, event="pulse", sequence=sequence, nonce=nonce, busy=busy,
                  failed_pci=failed, error=error, local_read_ns=closed_ns - opened_ns))
    require(not incoming.read(1), "unexpected trailing pulse input")


def worker(request, points, pins_file, pins_sha, parent_expected):
    node = socket.gethostname().split(".")[0]
    parent, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    require(HOST.fullmatch(node) and parent == parent_expected and parent in PARENTS
            and step.isdecimal(), "unexpected worker Slurm provenance")
    pins, actual = pin_json(pins_file)
    require(actual == pins_sha and node in pins, "worker PCI pins changed")
    run_worker(dict(node=node, parent=parent, step=f"{parent}.{step}", request=request, pci_pins_SHA256=pins_sha),
               pins[node], points, sys.stdin.buffer,
               lambda row: print(json.dumps(row, separators=(",", ":")), flush=True), Path("/sys/class/drm"))


class Receiver:
    def __init__(self, owners, request, pins_sha, pins):
        self.owners, self.request, self.pins_sha, self.pins = owners, request, pins_sha, pins
        self.ready, self.parent_steps = {}, {}
        self.chunks = defaultdict(bytearray)

    def accept(self, row, parent, event, into, arrival, packet=None, sent=None):
        node = row["node"]
        step = row["step"]
        require(node in self.owners and parent == row["parent"] == self.owners[node][0]
                and row["event"] == event and row["request"] == self.request
                and row["pci_pins_SHA256"] == self.pins_sha
                and re.fullmatch(parent + r"\.[0-9]+", step) and node not in into,
                "foreign worker, changed source or duplicate node reply")
        if event == "ready":
            require(self.parent_steps.setdefault(parent, step) == step, "multiple worker steps inside one parent")
        else:
            busy, failed = row["busy"], row["failed_pci"]
            require(node in self.ready and self.ready[node][0]["step"] == step
                    and type(row["sequence"]) is int and row["sequence"] == packet["sequence"]
                    and row["nonce"] == packet["nonce"], "pulse did not match ready worker, nonce and sequence")
            require(type(busy) is dict and type(failed) is list and len(failed) == len(set(failed))
                    and not set(busy).intersection(failed) and set(busy).union(failed) == set(self.pins[node])
                    and all(type(v) is int and 0 <= v <= 100 for v in busy.values())
                    and (row["error"] is None or row["error"] == "DRM_INVENTORY_UNVERIFIED" and not busy)
                    and type(row["local_read_ns"]) is int and 0 <= row["local_read_ns"] <= arrival - sent,
                    "invalid busy values, incomplete PCI denominator or worker interval")
        into[node] = row, arrival

    def receive(self, selector, event, into, deadline, packet=None, sent=None):
        while len(into) < len(self.owners):
            remaining = (deadline - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                raise TimeoutError("missing planned pulse replies")
            for key, _ in selector.select(remaining):
                data = os.read(key.fd, 65536)
                arrival = time.monotonic_ns()  # after all bytes in this read entered the controller
                require(data, "Slurm output closed before all expected replies")
                chunk = self.chunks[key.fd]
                chunk.extend(data)
                require(len(chunk) <= 131072, "unbounded worker output")
                while b"\n" in chunk:
                    split = chunk.index(b"\n")
                    require(0 < split <= 4096, "worker reply has invalid extent")
                    row = json.loads(chunk[:split])
                    del chunk[:split + 1]
                    self.accept(row, key.data, event, into, arrival, packet, sent)
            require(all(len(chunk) <= 4096 for chunk in self.chunks.values()), "unbounded worker record")


def summarize(owners, pins, pulses, threshold_ns, accounting_ok):
    require_pins(owners, pins)
    expected_cards = sum(len(cards) for cards in pins.values())
    output = []
    all_busy, all_cards_observed = [], set()
    for seq, pulse in enumerate(pulses):
        sent, rows = pulse["sent_ns"], pulse["rows"]
        require(set(rows) <= set(owners) and (sent is not None or not rows), "point has invalid controller bounds")
        arrivals = [arrival for _, arrival in rows.values()]
        require(all(0 < sent < arrival for arrival in arrivals), "controller time reversed")
        span = max(arrivals) - sent if arrivals else None
        missing = sorted(set(owners) - set(rows))
        incomplete = {node: sorted(set(pins[node]) - set(rows[node][0]["busy"]) if node in rows else pins[node])
                      for node in owners if node not in rows or len(rows[node][0]["busy"]) != 8}
        point_busy = [value for row, _ in rows.values() for value in row["busy"].values()]
        all_busy.extend(point_busy)
        all_cards_observed.update((node, pci) for node, (row, _) in rows.items() for pci in row["busy"])
        complete = span is not None and span <= threshold_ns and not incomplete and len(point_busy) == expected_cards
        output.append(dict(sequence=seq, nonce=pulse.get("nonce"), expected_nodes=len(owners), expected_samples=expected_cards,
                           responded_nodes=len(rows), missing_nodes=missing,
                           missing_PCI_by_node=incomplete, valid_samples=len(point_busy),
                           sample_span_upper_bound_ns=span, controller_send_monotonic_ns=sent,
                           controller_last_receipt_monotonic_ns=max(arrivals) if arrivals else None,
                           complete=complete,
                           fleet_mean_busy_percent=sum(point_busy) / len(point_busy) if complete and accounting_ok else None,
                           nonzero_cards=sum(value > 0 for value in point_busy) if complete and accounting_ok else None,
                           max_busy_percent=max(point_busy) if complete and accounting_ok else None,
                           worker_reports=[dict(row, controller_receipt_monotonic_ns=arrival)
                                           for row, arrival in (rows[node] for node in sorted(rows))]))
    complete = accounting_ok and all(point["complete"] for point in output)
    return dict(complete=complete, planned_nodes=len(owners), expected_unique_node_PCI=expected_cards,
                observed_unique_node_PCI=len(all_cards_observed),
                expected_samples=expected_cards * len(pulses), valid_samples=len(all_busy),
                missing_samples=expected_cards * len(pulses) - len(all_busy), sample_span_threshold_ns=threshold_ns,
                planned_pulse_interval_ns=INTERVAL_NS,
                fleet_mean_busy_percent=sum(all_busy) / len(all_busy) if complete else None,
                all_fleet_points_strictly_above_80_percent=(
                    all(point["fleet_mean_busy_percent"] > 80 for point in output) if complete else None),
                points=output, clock="controller_monotonic_send_to_last_receipt_upper_bound",
                counter="sysfs_gpu_busy_percent", GPU_requested=0, sustained_interval_qualified=False)


def query(argv, deadline):
    seconds = (deadline - time.monotonic_ns()) / 1e9
    require(seconds > 0, "scheduler evidence deadline exhausted")
    done = subprocess.run(argv, capture_output=True, timeout=min(5, seconds), check=False)
    require(done.returncode == 0 and len(done.stdout) <= 262144 and len(done.stderr) <= 65536,
            "scheduler evidence command failed")
    return done.stdout.decode("utf-8").strip().splitlines()


def hostnames(expression, deadline):
    names = query(["scontrol", "show", "hostnames", expression], deadline)
    require(len(names) == len(set(names)), "duplicate node in scheduler placement")
    return set(names)


def check_live_parents(expected, deadline):
    rows = query(["squeue", "-h", "-j", ",".join(sorted(expected)), "-o", "%i|%T|%D|%N"], deadline)
    seen = set()
    for line in rows:
        parent, state, size, expression = line.split("|")
        require(parent in expected and parent not in seen and state == "RUNNING" and size == "40"
                and hostnames(expression, deadline) == set(expected[parent]), "live parent allocation changed")
        [owned] = query(["scontrol", "show", "job", parent, "-o"], deadline)
        fields = dict(part.split("=", 1) for part in owned.split() if "=" in part)
        require(fields.get("JobId") == parent and fields.get("JobState") == "RUNNING"
                and fields["UserId"].endswith(f"({os.getuid()})")
                and "gres/dcu=320" in fields["AllocTRES"].split(","), "parent ownership or GPU allocation differs")
        seen.add(parent)
    require(seen == set(expected), "live parent allocation absent")


def account(parent, label, deadline):
    rows = query(["sacct", "-n", "-P", "-j", parent, "--format=" + ",".join(COLUMNS)], deadline)
    selected = []
    for line in rows:
        row = line.split("|")
        require(len(row) == len(COLUMNS), "scheduler accounting row malformed")
        if row[1] == label:
            require(re.fullmatch(parent + r"\.[0-9]+", row[0]), "accounting step is foreign")
            selected.append(row)
    require(len(selected) <= 1, "ambiguous original step")
    return selected[0] if selected else None


def terminal(rows):
    return all(row and row[2].split() and row[2].split()[0] in TERMINAL for row in rows.values())


def qualified_accounting(rows, worker_steps, expected, get_hosts):
    if not expected or not set(expected) <= PARENTS or set(rows) != set(expected) or set(worker_steps) != set(expected) or not terminal(rows):
        return False
    for parent, row in rows.items():
        if row[0] != worker_steps[parent] or row[2:4] != ["COMPLETED", "0:0"]:
            return False
        fields = [field.split("=", 1) for field in row[6].split(",")]
        if any(len(field) != 2 for field in fields):
            return False
        resources = dict(fields)
        if (len(resources) != len(fields) or any(key.startswith("gres/") for key in resources)
                or resources.get("cpu") != "40" or resources.get("node") != "40"
                or resources.get("mem") not in {"10G", "10240M"} or get_hosts(row[5]) != set(expected[parent])):
            return False
    return True


def slurm_argv(parent, nodes, label, request, points, pins_path, pins_sha):
    return ["srun", "--jobid=" + parent, "--job-name=" + label, "--exact", "--overlap", "--immediate=5",
            "--nodes=40", "--ntasks=40", "--ntasks-per-node=1", "--cpus-per-task=1", "--mem=256M",
            "--gres=none", "--time=00:01:00", "--kill-on-bad-exit=0", "--input=all",
            "--nodelist=" + ",".join(nodes), PYTHON, "-E", "-s", "-S", "-B", "-u",
            str(Path(__file__).resolve()), "worker",
            "--request", request, "--points", str(points), "--pci-pins", str(pins_path),
            "--pins-sha", pins_sha, "--parent", parent]


def collect(ledger_path, pins_path, request, points, threshold_ns, output):
    ledger_raw, ledger_sha = pin_bytes(ledger_path)
    pins, pins_sha = pin_json(pins_path)
    _, groups, owners = allocation(ledger_raw)
    require_pins(owners, pins)
    require(all(HOST.fullmatch(node) for node in owners), "invalid node label")
    expected = {p: sorted(node for node, owner in owners.items() if owner[0] == p) for p in sorted(groups)}
    labels = {p: f"training80-busy-pulse-{request}-{p}" for p in expected}
    origin = time.monotonic_ns()
    origin_utc = datetime.now(timezone.utc).isoformat()
    readiness_deadline = origin + 30_000_000_000
    work_deadline = origin + 48_000_000_000
    output.mkdir(mode=0o700)
    commands = {p: slurm_argv(p, nodes, labels[p], request, points, pins_path, pins_sha)
                for p, nodes in expected.items()}
    store(output / "intent.json", dict(request=request, expected=expected, labels=labels, commands=commands,
                                        ledger_SHA256=ledger_sha, pci_pins_SHA256=pins_sha,
                                        sample_span_threshold_ns=threshold_ns, planned_points=points,
                                        GPU_requested=0, timed_pulse_interval_ns=INTERVAL_NS,
                                        controller_started_utc=origin_utc))
    receiver = Receiver(owners, request, pins_sha, pins)
    children, errors, accounting = {}, [], dict.fromkeys(expected)
    pulses = [dict(sent_ns=None, rows={}) for _ in range(points)]
    phase = "admission"
    with selectors.DefaultSelector() as selector:
        try:
            check_live_parents(expected, readiness_deadline)
            require(all(account(p, label, readiness_deadline) is None for p, label in labels.items()),
                    "request already has a numbered Slurm step")
            phase = "sampling"
            for parent, argv in commands.items():
                with (output / (parent + ".stderr")).open("xb") as stderr:
                    child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
                children[parent] = child
                selector.register(child.stdout, selectors.EVENT_READ, parent)
            receiver.receive(selector, "ready", receiver.ready, readiness_deadline)
            previous = None
            for seq, point in enumerate(pulses):
                if previous is not None:
                    time.sleep(max(0, (previous + INTERVAL_NS - time.monotonic_ns()) / 1e9))
                require(time.monotonic_ns() + REPLY_WAIT_NS <= work_deadline, "no budget for another pulse")
                packet = dict(request=request, sequence=seq, nonce=uuid4().hex)
                wire = json.dumps(packet, separators=(",", ":")).encode() + b"\n"
                point["nonce"] = packet["nonce"]
                previous = point["sent_ns"] = time.monotonic_ns()
                for child in children.values():
                    child.stdin.write(wire)
                    child.stdin.flush()
                receiver.receive(selector, "pulse", point["rows"], previous + REPLY_WAIT_NS, packet, previous)
            require(not any(receiver.chunks.values()), "trailing partial worker output")
            for child in children.values():
                child.stdin.close()
            for child in children.values():
                require(child.wait(timeout=max(.01, (work_deadline - time.monotonic_ns()) / 1e9)) == 0,
                        "nonzero Slurm sampler exit")
            end = selector.select(max(0, min(.2, (work_deadline - time.monotonic_ns()) / 1e9)))
            require(len(end) == len(children) and all(os.read(key.fd, 1) == b"" for key, _ in end),
                    "worker output did not close after planned pulses")
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as err:
            errors.append({"phase": phase, "type": type(err).__name__})
        finally:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
                if not child.stdin.closed:
                    try:
                        child.stdin.close()
                    except BrokenPipeError:
                        pass
                child.stdout.close()
    qualified = False
    try:
        accounting_deadline = time.monotonic_ns() + 12_000_000_000
        while True:
            for parent, label in labels.items():
                accounting[parent] = account(parent, label, accounting_deadline)
            if terminal(accounting) or time.monotonic_ns() + 1_000_000_000 >= accounting_deadline:
                break
            time.sleep(.5)
        qualified = qualified_accounting(accounting, receiver.parent_steps, expected,
                                         lambda expression: hostnames(expression, accounting_deadline))
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as err:
        errors.append({"phase": "accounting", "type": type(err).__name__})
    result = summarize(owners, pins, pulses, threshold_ns, qualified and not errors)
    result.update(request=request, actual_accounting=accounting, accounting_columns=list(COLUMNS),
                  accounting_qualified=qualified, failures=errors, ready_nodes=len(receiver.ready),
                  ledger_SHA256=ledger_sha, pci_pins_SHA256=pins_sha, controller_started_utc=origin_utc,
                  controller_finished_utc=datetime.now(timezone.utc).isoformat())
    store(output / "outcome.json", result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("collect", "worker"))
    cli.add_argument("--request", required=True)
    cli.add_argument("--points", type=int, choices=(3, 5), default=3)
    cli.add_argument("--pci-pins", type=Path, required=True)
    cli.add_argument("--ledger", type=Path)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--max-span-ms", type=int, default=MAX_LOCAL_NS // 1_000_000)
    cli.add_argument("--pins-sha")
    cli.add_argument("--parent")
    args = cli.parse_args()
    require(token(args.request, 32), "request must be a fresh 32-hex identity")
    if args.mode == "worker":
        worker(args.request, args.points, args.pci_pins, args.pins_sha, args.parent)
        return 0
    require(args.ledger and args.output_dir and 1 <= args.max_span_ms <= 1000, "invalid controller scope")
    result = collect(args.ledger.resolve(strict=True), args.pci_pins.resolve(strict=True), args.request,
                     args.points, args.max_span_ms * 1_000_000, args.output_dir)
    bounded = dict(result, points=[{k: v for k, v in point.items() if k not in
                                  {"worker_reports", "missing_PCI_by_node"}} for point in result["points"]])
    print(json.dumps(bounded, sort_keys=True, allow_nan=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
