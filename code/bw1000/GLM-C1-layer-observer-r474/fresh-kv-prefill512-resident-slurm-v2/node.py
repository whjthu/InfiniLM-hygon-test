"""Capture host-side physical state and enter the verified model container."""

import os
import json
from pathlib import Path
import socket
import sys
import time

from contract import CONSTRUCT, HERE, NODES, PORTS, WHOLE, budget, need, proc_start, put, read_json, sha, step_identity


def snapshot_mode(mode, intent, identify=step_identity):
    from snapshot import drm_inventory, gpu_holders, kfd_inventory

    node = socket.gethostname().split(".")[0]
    need(node in NODES, "node differs from concrete I02 claim")
    address = intent["claim"]["data_IPs"][NODES.index(node)]
    host_ips = {info[4][0] for info in socket.getaddrinfo(node, None, socket.AF_INET, socket.SOCK_STREAM)}
    need(address in host_ips, "fixed data IP does not match actual host resolver")
    for port in PORTS:
        with socket.socket() as probe:
            probe.bind((address, port))
    pci = drm_inventory()
    holders, kernel_holders, denied = gpu_holders()
    need(pci == intent["claim"]["PCI_by_node"][node] and holders == kernel_holders == denied == 0,
         "pinned physical DRM PCI or holders differ")
    status,topology = read_topology(mode,pci,kfd_inventory)
    return dict(phase=mode, node=node, actual_step=identify(), address=address, ports=PORTS, PCI=pci,
                KFD_topology_status=status,
                PCI_to_KFD=topology, holders=holders, kernel_holders=kernel_holders, unreadable_FD=denied,
                boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                monotonic_ns=time.monotonic_ns())


def validate_topology(pci, topology):
    need(type(topology) is dict and len(pci) == len(topology) == len(set(pci)) == 8
         and set(topology) == set(pci) and len(set(topology.values())) == 8
         and all(type(v) is int and v > 0 for v in topology.values()), "leased KFD/DRM topology differs")


def read_topology(mode,pci,reader):
    if mode in {"pre","post"}:
        # Actual zero-GRES diagnostics showed the unallocated topology read is
        # denied. Holder inspection remains separate and mandatory.
        return "UNAVAILABLE_ZERO_GRES",None
    need(mode == "device", "unexpected GPU topology phase")
    mapping = reader()
    validate_topology(pci,mapping)
    return "VERIFIED_LEASED",mapping


def hardware_matches(before, later, *, gate=False):
    need(all(before[k] == later[k] for k in ("node", "address", "boot", "PCI", "ports"))
         and later["holders"] == later["kernel_holders"] == later["unreadable_FD"] == 0,
         "physical identity changed or resource still held")
    need(before["KFD_topology_status"] == "UNAVAILABLE_ZERO_GRES" and before["PCI_to_KFD"] is None,
         "original zero-GRES preflight cannot assert topology")
    if gate:
        need(later["KFD_topology_status"] == "VERIFIED_LEASED", "missing prior leased KFD qualification")
        validate_topology(later["PCI"],later["PCI_to_KFD"])
    else:
        need(later["KFD_topology_status"] == "UNAVAILABLE_ZERO_GRES" and later["PCI_to_KFD"] is None,
             "zero-GRES evidence cannot assert topology")
    delta = later["monotonic_ns"] - before["monotonic_ns"]
    need(delta >= 0 and (not gate or delta <= 90_000_000_000), "fresh per-node clock/boot evidence differs")


def verify_post_binding(base,post,leased,terminal_step):
    hardware_matches(base,post)
    if leased is None:
        return None
    need(leased["actual_step"] == terminal_step and leased["phase"] == "device", "foreign leased PCI receipt")
    hardware_matches(base,leased,gate=True)
    need(post["monotonic_ns"] >= leased["monotonic_ns"], "cleanup preceded the leased observation")
    return leased["actual_step"]


def read_lease(out,node,step):
    base = read_json(out/f"pre-{node}.json")
    lease = read_json(out/f"device-{node}.json")
    need(lease["node"] == node and lease["phase"] == "device" and lease["actual_step"] == step,
         "foreign GPU admission receipt")
    hardware_matches(base,lease,gate=True)
    return lease


def wait_both_leases(out,step,epoch):
    """Both HOST tasks prove eight leased cards before either maps a provider."""
    deadline = time.monotonic()+min(35,budget(epoch,CONSTRUCT))
    joined = {}
    while len(joined) < len(NODES) and time.monotonic() < deadline:
        for node in NODES:
            if node in joined:
                continue
            try:
                entry = read_json(out/f"device-entry-{node}.json")
            except (FileNotFoundError,json.JSONDecodeError):
                continue
            need(entry["node"] == node and entry["actual_step"] == step, "foreign GPU entrance disposition")
            need(entry["state"] == "VERIFIED", "one GPU node rejected leased KFD admission")
            try:
                joined[node] = read_lease(out,node,step)
            except (FileNotFoundError,json.JSONDecodeError):
                continue
        if len(joined) < len(NODES):
            time.sleep(.05)
    need(set(joined) == set(NODES), "incomplete two-node leased KFD admission")
    return joined


def reconcile_processes(folder, node, gpu_step, start=proc_start):
    staged = []
    index = NODES.index(node)
    paths = [folder / f"leader-{index}.json", *(folder / f"owner-{i:02d}.json" for i in range(index*8,index*8+8))]
    for path in paths:
        if not path.is_file():
            continue
        report = read_json(path)
        need(report["identity"]["node"] == node and report["run"]["step"] == gpu_step,
             "foreign node or step in saved process identity")
        for item in [report["identity"], *report.get("spawned", [])]:
            pid, ticks = item["PID"], item["startticks"]
            need(type(pid) is int and type(ticks) is int and pid > 0 and ticks > 0, "invalid exact owned PID")
            try:
                actual = start(pid)
            except FileNotFoundError:
                actual = None
            need(actual != ticks, "original process still live")
            if (pid,ticks) not in staged:
                staged.append((pid,ticks))
    return staged


def main(mode, out):
    root = Path(out)
    need(mode in {"pre", "device", "post"}, "unexpected host probe action")
    intent = read_json(root / "intent.json")
    budget(intent["epoch"], WHOLE)
    if mode == "device":
        node = socket.gethostname().split(".")[0]
        step = step_identity()
        need(node in NODES,"node differs from new device scope")
        entrance = root/f"device-entry-{node}.json"
        try:
            record = snapshot_mode(mode,intent)
            hardware_matches(read_json(root/f"pre-{node}.json"),record,gate=True)
            put(root/f"device-{node}.json",record)
        except Exception as exc:
            put(entrance,dict(node=node,actual_step=step,state="REJECTED",reason_type=type(exc).__name__,
                              errno=getattr(exc,"errno",None),reason=str(exc)[:256]))
            raise
        put(entrance,dict(node=node,actual_step=step,state="VERIFIED"))
        wait_both_leases(root,step,intent["epoch"])
        os.execv("/bin/bash", ["bash", str(HERE/"container.sh"),str(root)])
        return
    record = snapshot_mode(mode, intent)
    node = record["node"]
    if mode == "post":
        terminal = read_json(root/"device-receipt.json")
        need(terminal["state"] == "TERMINAL" and terminal["row"][0] != record["actual_step"],
             "post requires distinct original terminal GPU step")
        step = terminal["row"][0]
        lease = root/f"device-{node}.json"
        leased = read_json(lease,4096) if lease.is_file() else None
        record["prior_leased_KFD_step"] = verify_post_binding(read_json(root/f"pre-{node}.json"),record,
                                                             leased,step)
        record["prior_leased_KFD_SHA256"] = sha(lease.read_bytes()) if leased else None
        record["checked_original_processes"] = reconcile_processes(root,node,step)
    put(root/f"{mode}-{node}.json",record)


if __name__ == "__main__":
    main(*sys.argv[1:])
