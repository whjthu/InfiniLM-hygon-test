"""Distribute the exact world and TP IDs used by GlobalTP16Context."""

import ctypes
import hashlib
import re
import time

from bootstrap import PORT, UID_BYTES, exchange
from verify_transport import NETWORK, mapped_providers


def distribute_ids(api, rank, address, step, run_manifest_sha, key_hex, seconds, mark, *, port=PORT):
    if not re.fullmatch(r"[0-9]+\.[0-9]+", step) or not re.fullmatch(r"[0-9a-f]{64}", run_manifest_sha):
        raise ValueError("invalid model step or manifest binding")
    if not re.fullmatch(r"[0-9a-f]{64}", key_hex) or not 0 < seconds <= 600:
        raise ValueError("invalid private bootstrap key or timeout")
    size = int(api["id_size"]())
    if size != UID_BYTES:
        raise ValueError("unexpected Native communicator ID size")

    def make_ids():
        ids = [ctypes.create_string_buffer(size) for _ in range(2)]
        mark("get_id.enter")
        for buffer in ids:
            if api["id"](buffer, size) != 0:
                raise RuntimeError("Native communicator ID creation failed")
        mark("get_id.return")
        if ids[0].raw == ids[1].raw:
            raise ValueError("world and TP IDs must be distinct")
        return ids[0].raw + ids[1].raw

    frame = exchange(rank, address, port, bytes.fromhex(key_hex), f"{step}:{run_manifest_sha}",
                     time.monotonic() + seconds, make_ids, mark, uid_bytes=2 * size)
    ids = (frame[:size], frame[size:])
    if ids[0] == ids[1]:
        raise ValueError("world and TP IDs must be distinct")
    return ids, [hashlib.sha256(value).hexdigest() for value in ids]


def model_transport_provider(mode, environ, maps):
    if mode not in NETWORK or environ.get("NCCL_NET") != NETWORK[mode]:
        raise ValueError("wrong model transport selection")
    expected = ("shca", "0") if mode == "shca" else ("none", "1")
    if (environ.get("NCCL_NET_PLUGIN"), environ.get("NCCL_IB_DISABLE")) != expected:
        raise ValueError("wrong model transport plugin policy")
    return mapped_providers(maps, mode)
