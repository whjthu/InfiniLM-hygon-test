"""Capture host-side physical state and enter the verified model container."""

import os
from pathlib import Path
import socket
import sys
import time

from contract import HERE, NODES, PORTS, WHOLE, budget, need, proc_start, put, read_json, step_identity


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
    topology = kfd_inventory()
    holders, kernel_holders, denied = gpu_holders()
    need(set(topology) == set(pci) and holders == kernel_holders == denied == 0, "physical devices or holders differ")
    return dict(phase=mode, node=node, actual_step=identify(), address=address, ports=PORTS, PCI=pci,
                PCI_to_KFD=topology, holders=holders, kernel_holders=kernel_holders, unreadable_FD=denied,
                boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                monotonic_ns=time.monotonic_ns())


def hardware_matches(before, later, *, gate=False):
    need(all(before[k] == later[k] for k in ("node", "address", "boot", "PCI", "PCI_to_KFD", "ports"))
         and later["holders"] == later["kernel_holders"] == later["unreadable_FD"] == 0,
         "physical identity changed or resource still held")
    delta = later["monotonic_ns"] - before["monotonic_ns"]
    need(delta >= 0 and (not gate or delta <= 90_000_000_000), "fresh per-node clock/boot evidence differs")


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
    record = snapshot_mode(mode, intent)
    node = record["node"]
    if mode != "pre":
        hardware_matches(read_json(root/f"pre-{node}.json"), record, gate=mode=="device")
    if mode == "post":
        terminal = read_json(root/"device-receipt.json")
        need(terminal["state"] == "TERMINAL" and terminal["row"][0] != record["actual_step"],
             "post requires distinct original terminal GPU step")
        record["checked_original_processes"] = reconcile_processes(root, node, terminal["row"][0])
    put(root/f"{mode}-{node}.json",record)
    if mode == "device":
        os.execv("/bin/bash", ["bash", str(HERE/"container.sh"),str(root)])


if __name__ == "__main__":
    main(*sys.argv[1:])
