"""Measure name and content visibility using the original two-node container mounts."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

root = Path(sys.argv[1])
rank = int(os.environ["SLURM_PROCID"])
node = socket.gethostname().split(".")[0]
step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
assert rank in (0, 1) and node == ("f13r1n08", "f13r1n09")[rank]
assert step.startswith("1103606.") and os.environ["SLURM_NTASKS"] == "2"
epoch = float(sys.argv[2])
test_bytes = hashlib.sha512(b"synthetic shared visibility probe").digest()
new_name = root / "published.bin"
primed = root / "primed.bin"
fixed_name = root / "existing.bin"
original = Path("/refactor/results/GLM474-TP16-transport-pair-v1/arm-1/uid.bin")

if rank == 0:
    # Give the reader time to observe the missing paths before creating them.
    time.sleep(max(0.0, epoch + 22 - time.time()))
    start_ns = time.time_ns()
    assert not new_name.exists() and not primed.exists()
    (root / "atomic.tmp").write_bytes(test_bytes)
    (root / "atomic.tmp").replace(new_name)
    with primed.open("xb") as stream:
        stream.write(test_bytes)
    with fixed_name.open("r+b") as stream:
        stream.write(test_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(published_epoch_ns=start_ns, complete_epoch_ns=time.time_ns())
else:
    first = {}
    attempts = 0
    started_ns = time.time_ns()
    limit = time.monotonic() + min(38, max(0, epoch + 60 - time.time()))
    while time.monotonic() < limit and len(first) < 3:
        attempts += 1
        if "rename_is_file" not in first and new_name.is_file():
            assert new_name.read_bytes() == test_bytes
            first["rename_is_file"] = time.time_ns()
        if "create_direct_open" not in first:
            try:
                data = primed.read_bytes()
            except FileNotFoundError:
                data = b""
            if data:
                assert data == test_bytes
                first["create_direct_open"] = time.time_ns()
        if "existing_content" not in first:
            data = fixed_name.read_bytes()
            if data == test_bytes:
                first["existing_content"] = time.time_ns()
            else:
                assert data == bytes(len(test_bytes))
        if len(first) < 3:
            time.sleep(.1)
    result = dict(reader_started_epoch_ns=started_ns, first_observed_epoch_ns=first, attempts=attempts,
                  reader_started_before_expected_publication=started_ns < int((epoch + 22) * 1e9))
st = original.stat()
assert st.st_size == 136
result.update(rank=rank, node=node, step=step, original_uid_size=st.st_size, original_uid_mtime_ns=st.st_mtime_ns)
with (root / f"rank-{rank}.json").open("x") as stream:
    json.dump(result, stream)
