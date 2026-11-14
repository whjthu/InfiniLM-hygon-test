import json
import os
from pathlib import Path
import socket
import sys

from snapshot import snapshot, require


def check_owner(node_index, pid, start):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return
    require(raw.rsplit(") ",1)[1].split()[19]!=str(start),f"rank or leader still live on node {node_index}")


def main(phase, folder):
    require(phase in ("pre","gate","post"),"unknown hardware probe phase")
    row = snapshot(phase=="gate")
    node_index = ("f13r1n05","f13r1n07").index(row["node"])
    require(node_index==int(os.environ["SLURM_PROCID"]),"node assignment changed")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0",43844))
        probe.listen(1)
    row["port_bindable"] = True
    require(row["kfd_holder_count"]==0,"live device owners")
    if phase=="gate":
        before = json.loads((folder/"pre"/(row["node"]+".json")).read_bytes())
        require(before["boot_id"]==row["boot_id"] and before["pci"]==row["pci"] and
                0<=row["monotonic_ns"]-before["monotonic_ns"]<=90_000_000_000,"stale or changed pre-run inventory")
    if phase=="post":
        leader = folder/f"leader-{node_index}.json"
        if leader.is_file():
            proof = json.loads(leader.read_bytes())
            check_owner(node_index,proof["PID"],proof["startticks"])
            for pid,start in proof.get("owners",[]):
                check_owner(node_index,pid,start)
        for device in range(8):
            trial = folder/f"owner-{node_index*8+device:02}.json"
            if trial.is_file():
                record = json.loads(trial.read_bytes())
                check_owner(node_index,record["owner_PID"],record["startticks"])
    with (folder/phase/(row["node"]+".json")).open("x") as stream:
        json.dump(row,stream,indent=2)
    if phase=="gate":
        os.execv("/bin/bash",["bash",str(Path(__file__).with_name("container.sh")),str(folder)])


if __name__=="__main__":
    main(sys.argv[1],Path(sys.argv[2]))
