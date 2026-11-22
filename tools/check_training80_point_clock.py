"""Measure point bounds on node wall-clock offsets using a common Slurm fan-out."""

import argparse
from collections import defaultdict
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


PARENTS = {"1087656", "1103606"}
HOST = re.compile(r"[a-z][a-z0-9]+")
MAX_LOCAL_NS = 250_000_000


def require(value, detail):
    if not value:
        raise ValueError(detail)


def point_bound(sent_ns, received_ns, remote_ns):
    """The remote observation follows send and precedes receive."""
    require(type(sent_ns) is type(received_ns) is type(remote_ns) is int
            and 0 < sent_ns < received_ns and remote_ns > 0, "invalid point-clock boundary")
    return [remote_ns - received_ns, remote_ns - sent_ns]


def bounded_slice(records):
    """Envelope for this pulse only. No interpolated clock drift or offset."""
    return dict(nodes=len(records), reference_roundtrip_max_ns=max(r["rtt_ns"] for r in records),
                global_relative_lower_ns=min(r["offset_ns"][0] for r in records),
                global_relative_upper_ns=max(r["offset_ns"][1] for r in records),
                node_time_skew_upper_ns=max(r["offset_ns"][1] for r in records)
                - min(r["offset_ns"][0] for r in records))


def worker():
    node = socket.gethostname().split(".")[0]
    parent, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    require(HOST.fullmatch(node) and parent in PARENTS and step.isdecimal(), "unexpected Slurm worker")
    print(json.dumps(dict(event="ready", node=node, parent=parent, step=parent + "." + step)), flush=True)
    for seq in range(3):
        line = sys.stdin.buffer.readline(512)
        observed = time.time_ns()
        packet = json.loads(line)
        require(type(packet["sequence"]) is int and packet["sequence"] == seq
                and re.fullmatch("[a-f0-9]{32}", packet["nonce"]), "unexpected or duplicate pulse")
        print(json.dumps(dict(event="pulse", node=node, parent=parent, step=parent + "." + step,
                              sequence=seq, nonce=packet["nonce"], wall_ns=observed,
                              monotonic_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))), flush=True)
    require(not sys.stdin.buffer.read(1), "unexpected trailing pulses")


def wall_sample():
    first = time.monotonic_ns()
    wall = time.time_ns()
    last = time.monotonic_ns()
    return first, wall, last


def interval(earlier, later):
    first, sent, sent_done = earlier
    received_first, received, last = later
    require(first <= sent_done <= received_first <= last, "controller monotonic order reversed")
    lo, hi = received_first - sent_done, last - first
    require(0 < received - sent and lo - 2_000_000 <= received - sent <= hi + 2_000_000,
            "controller wall discontinuity at this pulse")
    return sent, received, hi


def collect(expected, worker_source, request):
    require(re.fullmatch("[a-f0-9]{32}", request) and set(expected) == PARENTS, "unexpected clock scope")
    require(all(nodes and len(nodes) == len(set(nodes)) and all(HOST.fullmatch(n) for n in nodes)
                for nodes in expected.values()), "bad clock placement")
    members = {node: parent for parent, nodes in expected.items() for node in nodes}
    require(len(members) == sum(map(len, expected.values())) <= 80, "duplicate node across parents")
    loops = selectors.DefaultSelector()
    children, chunks, seen = {}, defaultdict(bytearray), {}
    bound = time.monotonic() + 40

    def receive(event):
        rows = {}
        while len(rows) < len(members):
            remain = bound - time.monotonic()
            if remain <= 0:
                raise TimeoutError("Slurm point clock missed planned node replies")
            for key, _ in loops.select(remain):
                data = os.read(key.fd, 65536)
                arrival = wall_sample()
                require(data, "unexpected end of Slurm pulse output")
                chunk = chunks[key.fd]
                chunk.extend(data)
                require(len(chunk) <= 131072, "clock probe exceeded line bound")
                while b"\n" in chunk:
                    sep = chunk.index(b"\n")
                    row = json.loads(chunk[:sep])
                    del chunk[:sep+1]
                    node, parent = row["node"], row["parent"]
                    require(row["event"] == event and members.get(node) == parent == key.data
                            and re.fullmatch(parent + r"\.[0-9]+", row["step"]) and node not in rows,
                            "unexpected clock worker or duplicate reply")
                    if event == "ready":
                        seen[node] = row["step"]
                    else:
                        require(seen[node] == row["step"], "clock worker step changed")
                    rows[node] = row, arrival
        return rows

    try:
        for parent, nodes in sorted(expected.items()):
            argv = ["srun", "--jobid=" + parent, "--exact", "--immediate=5",
                    "--nodes=" + str(len(nodes)), "--ntasks=" + str(len(nodes)), "--ntasks-per-node=1",
                    "--cpus-per-task=1", "--mem=256M", "--gres=none", "--time=00:01:00",
                    "--input=all", "--nodelist=" + ",".join(nodes), "--job-name=training80-point-clock-" + request,
                    "/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-u", "-c", worker_source, "worker"]
            child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            children[parent] = child
            loops.register(child.stdout, selectors.EVENT_READ, parent)
        receive("ready")
        slices = []
        for seq in range(3):
            nonce = uuid4().hex
            pulse = (json.dumps(dict(sequence=seq, nonce=nonce)) + "\n").encode()
            anchor = wall_sample()
            for child in children.values():
                child.stdin.write(pulse)
                child.stdin.flush()
            replies = receive("pulse")
            records = []
            for node, (row, arrival) in sorted(replies.items()):
                require(row["sequence"] == seq and row["nonce"] == nonce
                        and type(row["monotonic_raw_ns"]) is int and row["monotonic_raw_ns"] > 0,
                        "clock reply missed pulse identity")
                sent, received, rtt = interval(anchor, arrival)
                records.append(dict(node=node, step=row["step"], rtt_ns=rtt, reference_send_ns=sent,
                                    reference_receive_ns=received, device_wall_ns=row["wall_ns"],
                                    offset_ns=point_bound(sent, received, row["wall_ns"])))
            slices.append(dict(sequence=seq, **bounded_slice(records), records=records))
        for child in children.values():
            child.stdin.close()
            require(child.wait(timeout=max(.1, bound-time.monotonic())) == 0, "nonzero Slurm clock exit")
        return dict(request=request, state="POINT_CLOCK_ENVELOPES_OBSERVED",
                    node_count=len(members), slices=slices,
                    all_points_within_250ms=all(p["reference_roundtrip_max_ns"] <= MAX_LOCAL_NS for p in slices),
                    controller_stable_wall_assumed_between_boundary_samples=True,
                    sustained_interval_qualified=False, GPU_requested=0)
    finally:
        loops.close()
        for child in children.values():
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "collect"))
    parser.add_argument("--request")
    parser.add_argument("--placement")
    args = parser.parse_args()
    if args.mode == "worker":
        worker()
    else:
        result = collect(json.loads(args.placement), Path(__file__).read_text(encoding="utf-8"), args.request)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
