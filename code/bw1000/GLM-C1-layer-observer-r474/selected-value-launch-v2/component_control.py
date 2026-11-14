"""Two Slurm leaders and sixteen local owners for the selected-value component."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from ready_socket_parent import canonical, receive_json, send_json
from runner import new_uids, run_owner

PARENT = "1103606"
NODES = ("f13r1n05", "f13r1n07")
IPS = ("10.6.13.5", "10.6.13.7")
PORT = 43844
WORK_SECONDS = 240
PHASES = ("before-create", "after-create", "numerics-passed", "before-close", "after-close")
HERE = Path(__file__).resolve().parent


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class Channel:
    def __init__(self, deadline, children=()):
        self.deadline = deadline
        self.children = children

    def stop_requested(self):
        return any(p.poll() not in (None, 0) for p in self.children)

    def send(self, sock, record):
        send_json(sock, record, self, lambda: False, self.deadline)

    def receive(self, sock):
        return receive_json(sock, self, lambda: False, self.deadline)

    def left(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("component work deadline")
        return left


def envelope(identity, phase, kind, **payload):
    return dict(identity=identity, phase=phase, kind=kind, **payload)


def validate_records(records, node, identity, phase):
    require(type(records) is list and len(records) == 8, "requires eight owner records")
    for device, row in enumerate(records):
        require(type(row) is dict and set(row) == {"identity", "phase", "rank", "pid", "startticks"}
                and row["identity"] == identity and row["phase"] == phase
                and type(row["rank"]) is int and row["rank"] == node*8 + device
                and type(row["pid"]) is int and row["pid"] > 0
                and type(row["startticks"]) is str and row["startticks"].isdecimal(),
                "invalid owner record")
    require(len({row["pid"] for row in records}) == 8, "local duplicate PID")
    return records


def gather(channel, peer, node, identity, phase, records):
    validate_records(records, node, identity, phase)
    local = envelope(identity, phase, "ready", records=records)
    if node == 0:
        channel.send(peer, local)
        remote = channel.receive(peer)
    else:
        remote = channel.receive(peer)
        channel.send(peer, local)
    require(type(remote) is dict and set(remote) == {"identity", "phase", "kind", "records"}
            and remote["identity"] == identity and remote["phase"] == phase
            and remote["kind"] == "ready", "wrong peer phase or execution identity")
    other = validate_records(remote["records"], 1-node, identity, phase)
    ordered = records + other if node == 0 else other + records
    digest = hashlib.sha256(canonical(ordered)).hexdigest()
    ack = envelope(identity, phase, "ack", digest=digest)
    if node == 0:
        channel.send(peer, ack)
        require(channel.receive(peer) == ack, "peer did not verify all 16 ranks")
    else:
        require(channel.receive(peer) == ack, "peer did not verify all 16 ranks")
        channel.send(peer, ack)
    return digest


def startticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def write_record(path, obj):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(canonical(obj)+b"\n")
    os.replace(tmp, path)


def child(args, identity, deadline):
    require(args.node in (0, 1) and args.device in range(8), "invalid child geometry")
    device = args.device
    mask = sorted(os.sched_getaffinity(0))
    require(len(mask) >= 64, "requires original node CPU allocation")
    os.sched_setaffinity(0, mask[4*device:4*device+4])
    path = args.out / f"owner-{args.node*8+device:02}.json"
    channel = Channel(deadline)
    with socket.socket(fileno=args.fd) as parent:
        packet = channel.receive(parent)
        require(isinstance(packet, dict) and set(packet) == {"identity", "world", "tp"}
                and packet["identity"] == identity, "invalid private CCL setup")
        world, tp = (base64.b64decode(packet[key], validate=True) for key in ("world", "tp"))
        def gate(phase, report):
            row = dict(identity=identity, phase=phase, rank=report["rank"],
                       pid=os.getpid(), startticks=startticks(os.getpid()))
            channel.send(parent, row)
            reply = channel.receive(parent)
            require(type(reply) is dict and set(reply) == {"identity", "phase", "kind", "digest"}
                    and reply["identity"] == identity and reply["phase"] == phase
                    and reply["kind"] == "GO" and re.fullmatch("[0-9a-f]{64}", reply["digest"]),
                    "invalid leader gate response")
        run_owner(args.node, device, world, tp, args.model, args.binding, gate,
                  lambda report: write_record(path, report))


def connect(node, channel):
    if node == 1:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((IPS[1], PORT))
            listener.listen(1)
            listener.settimeout(channel.left())
            conn, peer = listener.accept()
        if peer[0] != IPS[0]:
            conn.close()
            raise ValueError("unexpected control peer address")
        return conn
    while True:
        conn = socket.socket()
        try:
            conn.bind((IPS[0], 0))
            conn.settimeout(min(channel.left(), 5))
            conn.connect((IPS[1], PORT))
            return conn
        except ConnectionRefusedError:
            conn.close()
            time.sleep(min(channel.left(), 0.1))
        except BaseException:
            conn.close()
            raise


def leader(args, identity, deadline):
    children, pipes = [], []
    channel = Channel(deadline, children)
    proof = dict(identity=identity, node=args.node, phases=[], passed=False,
                 PID=os.getpid(), startticks=startticks(os.getpid()), owners=[])
    write_record(args.out/f"leader-{args.node}.json", proof)
    try:
        with connect(args.node, channel) as peer:
            # Opaque CCL IDs stay on the two allocated nodes and local socketpairs.
            if args.node == 0:
                world, tp = new_uids()
                packet = dict(identity=identity, world=base64.b64encode(world).decode("ascii"),
                              tp=base64.b64encode(tp).decode("ascii"))
                channel.send(peer, packet)
            else:
                packet = channel.receive(peer)
                require(type(packet) is dict and set(packet) == {"identity", "world", "tp"}
                        and packet["identity"] == identity, "unexpected CCL setup identity")
            for device in range(8):
                parent, owner = socket.socketpair()
                pipes.append(parent)
                rank = args.node*8+device
                with owner, (args.out/f"owner-{rank:02}.log").open("xb") as log:
                    argv = [sys.executable, "-B", str(__file__), "--node", str(args.node),
                            "--device", str(device), "--fd", str(owner.fileno()),
                            "--out", str(args.out), "--model", str(args.model),
                            "--binding", str(args.binding)]
                    p = subprocess.Popen(argv, pass_fds=(owner.fileno(),),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                    children.append(p)
                proof["owners"].append((p.pid,startticks(p.pid)))
                write_record(args.out/f"leader-{args.node}.json", proof)
                channel.send(parent, packet)
            del packet
            owners = proof["owners"]
            for phase in PHASES:
                records = [channel.receive(pipe) for pipe in pipes]
                validate_records(records, args.node, identity, phase)
                require([(r["pid"], r["startticks"]) for r in records] == owners,
                        "owner process identity changed")
                digest = gather(channel, peer, args.node, identity, phase, records)
                for pipe in pipes:
                    channel.send(pipe, envelope(identity, phase, "GO", digest=digest))
                proof["phases"].append(dict(phase=phase, all16_SHA256=digest))
                write_record(args.out/f"leader-{args.node}.json", proof)
            for p in children:
                require(p.wait(timeout=channel.left()) == 0, "rank owner returned failure")
            proof["passed"] = True
    finally:
        for pipe in pipes:
            pipe.close()
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)
        proof["exitcodes"] = [p.returncode for p in children]
        write_record(args.out/f"leader-{args.node}.json", proof)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--device", type=int)
    p.add_argument("--fd", type=int)
    args = p.parse_args()
    step = os.environ["SLURM_STEP_ID"]
    require(os.environ["SLURM_JOB_ID"] == PARENT and step.isdecimal()
            and args.node in (0, 1) and int(os.environ["SLURM_PROCID"]) == args.node
            and socket.gethostname().split(".")[0] == NODES[args.node], "wrong Slurm placement")
    claim = os.environ["GLM_SELECTED_CLAIM"]
    epoch = float(os.environ["GLM_SELECTED_EPOCH"])
    now = time.time()
    require(re.fullmatch("[0-9a-f]{40}", claim) and math.isfinite(epoch)
            and 0 <= now-epoch < WORK_SECONDS, "bad original execution identity/deadline")
    identity = dict(step=PARENT+"."+step, claim=claim, epoch=epoch)
    deadline = time.monotonic()+WORK_SECONDS-(now-epoch)
    if args.fd is None:
        leader(args, identity, deadline)
    else:
        child(args, identity, deadline)


if __name__ == "__main__":
    main()
