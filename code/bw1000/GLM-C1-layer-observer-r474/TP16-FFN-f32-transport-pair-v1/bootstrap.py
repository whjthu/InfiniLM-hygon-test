"""Bound the one-shot TP16 ID exchange to this Slurm step."""
import hashlib
import hmac
import json
import re
import socket
import sys
import time

PORT = 43863
RANKS = 16
UID_BYTES = 136


def remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("communicator ID exchange deadline expired")
    return left


def read_exact(sock, size, deadline):
    data = bytearray()
    while len(data) < size:
        sock.settimeout(remaining(deadline))
        part = sock.recv(size - len(data))
        if not part:
            raise ValueError("incomplete communicator ID exchange frame")
        data.extend(part)
    return bytes(data)


def send(sock, data, deadline):
    sock.settimeout(remaining(deadline))
    sock.sendall(data)


def configure_listener(sock):
    if sys.platform == "win32":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def summarize_stages(log):
    rank_zero = ("listener.ready", "get_id.enter", "get_id.return", "uid.acknowledged_all")
    rank_all = ("uid.ready", "comm_init.enter", "comm_init.return", "first_collective.return")
    stages = {name: [] for name in rank_zero + rank_all}
    digests = []
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(row, dict) or row.get("stage") not in stages:
            continue
        name = row["stage"]
        rank = row.get("rank")
        if type(rank) is not int or not 0 <= rank < RANKS:
            raise ValueError("invalid bootstrap stage rank")
        stages[name].append(rank)
        if name == "uid.ready":
            digest = row.get("uid_sha256")
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("invalid bootstrap ID digest")
            digests.append(digest)
    complete = (
        all(stages[name] == [0] for name in rank_zero)
        and all(sorted(stages[name]) == list(range(RANKS)) for name in rank_all)
        and len(set(digests)) == 1
    )
    return dict(stages=stages, complete=complete,
                uid_sha256=digests[0] if digests and len(set(digests)) == 1 else None)


def exchange(rank, address, port, key, step, deadline, make_uid, mark):
    if not 0 <= rank < RANKS or len(key) != 32:
        raise ValueError("invalid rank or bootstrap key")
    context = step.encode("ascii") + b":"
    def signature(data):
        return hmac.digest(key, context + data, "sha256")

    if rank == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            configure_listener(listener)
            listener.bind((address, port))
            listener.listen(RANKS - 1)
            mark("listener.ready")
            uid = make_uid()
            if len(uid) != UID_BYTES:
                raise ValueError("invalid native communicator ID length")
            seen = set()
            while len(seen) < RANKS - 1:
                listener.settimeout(remaining(deadline))
                client, _ = listener.accept()
                with client:
                    request = read_exact(client, 33, deadline)
                    peer = request[0]
                    if not 0 < peer < RANKS or peer in seen or not hmac.compare_digest(
                        request[1:], signature(b"request:" + request[:1])
                    ):
                        raise ValueError("invalid or duplicate bootstrap rank")
                    send(client, uid + signature(b"uid:" + request[:1] + uid), deadline)
                    ack = read_exact(client, 32, deadline)
                    if not hmac.compare_digest(ack, signature(b"ack:" + request[:1] + uid)):
                        raise ValueError("invalid bootstrap acknowledgement")
                    seen.add(peer)
            mark("uid.acknowledged_all")
    else:
        while True:
            try:
                client = socket.create_connection((address, port), timeout=min(1, remaining(deadline)))
                break
            except (ConnectionRefusedError, TimeoutError):
                time.sleep(min(.1, remaining(deadline)))
        with client:
            rank_byte = bytes([rank])
            send(client, rank_byte + signature(b"request:" + rank_byte), deadline)
            frame = read_exact(client, UID_BYTES + 32, deadline)
            uid = frame[:UID_BYTES]
            if not hmac.compare_digest(frame[UID_BYTES:], signature(b"uid:" + rank_byte + uid)):
                raise ValueError("invalid communicator ID response")
            send(client, signature(b"ack:" + rank_byte + uid), deadline)
    mark("uid.ready", uid_sha256=hashlib.sha256(uid).hexdigest())
    return uid
