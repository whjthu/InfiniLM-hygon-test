"""Host-side occupancy probe and in-allocation GPU ownership gate."""
import json
import os
from pathlib import Path
import socket
import sys
import subprocess
import time
from budget import PAIR_SECONDS, remaining

from snapshot import snapshot
from shared import PORT, require
from bootstrap import PORT as BOOTSTRAP_PORT, configure_listener


def probe(leased):
    row = snapshot(leased)
    ports = {}
    for port in (43843, PORT, BOOTSTRAP_PORT):
        with socket.socket() as sock:
            try:
                if port == BOOTSTRAP_PORT:
                    configure_listener(sock)
                sock.bind(("0.0.0.0", port))
                sock.listen(1)
                ports[str(port)] = True
            except OSError:
                ports[str(port)] = False
    row["ports_bindable"] = ports
    require(row["kfd_holder_count"] == 0 and row["same_uid_fd_denied_count"] == 0 and all(ports.values()), "node has live GPU or TCP ownership")
    return row


def verify_before(before, gate):
    require(before["node"] == gate["node"] and before["boot_id"] == gate["boot_id"]
            and before["pci"] == gate["pci"], "hardware identity changed")
    require(0 <= gate["monotonic_ns"] - before["monotonic_ns"] <= 90_000_000_000,
            "pre-run snapshot is stale")


def main(phase, folder):
    require(phase in {"pre", "gate", "post"}, "unknown phase")
    row = probe(phase == "gate")
    folder = Path(folder)
    if phase == "gate":
        before = json.loads((folder / "pre" / (row["node"] + ".json")).read_bytes())
        verify_before(before, row)
    if phase == "post":
        for file in [*folder.glob("worker-[01].json"), *folder.glob("owner-[0-9][0-9].json")]:
            owner = json.loads(file.read_bytes())
            if owner["node"] != row["node"]:
                continue
            records = [owner, *owner.get("owners", [])]
            for entry in records:
                pid = entry["PID"]
                require(type(pid) is int and pid > 0, "invalid original owner PID")
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
                except FileNotFoundError:
                    continue
                require(int(stat[19]) != int(entry["startticks"]), "original owner still live")
    if phase == 'pre':
        outer = Path(__file__).with_name('host_container.sh')
        left = min(135, remaining(float(os.environ['GLM470_EPOCH']), PAIR_SECONDS)-60)
        require(left > 0, 'preflight lacks original epoch budget')
        subprocess.run(['/bin/bash', str(outer), str(folder), 'preflight'], check=True, timeout=left)
        fresh = probe(False)
        require(fresh['pci'] == row['pci'] and fresh['boot_id'] == row['boot_id'], 'preflight hardware changed')
        row = fresh
    with (folder / phase / (row['node'] + '.json')).open('x') as stream:
        json.dump(row, stream, indent=2)
    if phase == "gate":
        outer = Path(__file__).with_name("host_container.sh")
        os.execv("/bin/bash", ["bash", str(outer), str(folder)])


if __name__ == "__main__":
    main(*sys.argv[1:])
