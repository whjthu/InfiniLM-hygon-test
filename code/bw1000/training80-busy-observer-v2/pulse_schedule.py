"""Keep a fixed-hour counter worker live across controller-side missed slots."""

import json
import os
from pathlib import Path
import socket
import sys
import time


def finish_packet(request, points):
    return {"request": request, "event": "finish", "scheduled_points": points}


def run_worker(short, identity, expected, points, incoming, send, drm, clock=time.monotonic_ns):
    short.require(len(expected) == len(set(expected)) == 8
                  and all(short.PCI.fullmatch(pci) for pci in expected), "invalid worker PCI pins")
    short.require(type(points) is int and points > 0, "invalid schedule")
    send(dict(identity, event="ready"))
    previous = -1
    seen = set()
    while True:
        raw = incoming.readline(512)
        short.require(raw.endswith(b"\n"), "incomplete observer schedule")
        packet = json.loads(raw)
        if (packet == finish_packet(identity["request"], points)
                and type(packet["scheduled_points"]) is int):
            short.require(not incoming.read(1), "unexpected trailing observer input")
            return
        short.require(isinstance(packet, dict) and set(packet) == {"request", "sequence", "nonce"}
                      and packet["request"] == identity["request"], "unexpected observer packet")
        sequence, nonce = packet["sequence"], packet["nonce"]
        short.require(type(sequence) is int and previous < sequence < points
                      and short.token(nonce, 32) and nonce not in seen, "invalid or duplicate observer pulse")
        previous = sequence
        seen.add(nonce)
        opened = clock()
        error = None
        try:
            busy, failed = short.read_counters(drm, expected)
        except (OSError, ValueError):
            busy, failed = {}, sorted(expected)
            error = "DRM_INVENTORY_UNVERIFIED"
        send(dict(identity, event="pulse", sequence=sequence, nonce=nonce, busy=busy,
                  failed_pci=failed, error=error, local_read_ns=clock() - opened))


def worker(short, request, points, pins_file, pins_sha, parent_expected):
    node = socket.gethostname().split(".")[0]
    parent, step = os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"]
    short.require(short.HOST.fullmatch(node) and parent == parent_expected
                  and parent in short.SUPPORTED_PARENTS and step.isdecimal(), "unexpected worker Slurm provenance")
    pins, actual = short.pin_json(pins_file)
    short.require(actual == pins_sha and node in pins, "worker PCI pins changed")
    identity = dict(node=node, parent=parent, step=f"{parent}.{step}", request=request, pci_pins_SHA256=pins_sha)
    run_worker(short, identity, pins[node], points, sys.stdin.buffer,
               lambda row: print(json.dumps(row, separators=(",", ":")), flush=True), Path("/sys/class/drm"))
