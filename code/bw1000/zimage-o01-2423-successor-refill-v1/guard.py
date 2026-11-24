"""Narrow append-only continuation of the ended O01 .2423 standby."""
import ast
from base64 import b64decode
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import time

NODE, PARENT = "f12r2n11", "1103606"
ADMISSION = "69fb6ba8d0e94fa39f90a52df3a5878c"
REQUEST = "ea2b693489164cb199a95fb844a9e243"
OLD = "274608fc6e0d410a8f545287843cb887"
PREVIOUS = "e23d24ea0c9c4d00af06154d82dc22e8"
PRIOR_REFILL = "3f96a7be936a4a018ff9f83623895428"
PRIOR_GRANT_SHA = "fdb2034de3494c58efc9d8f1f6cbc712ac5e847047fac9192284dedffb77abcb"
PRIOR_WORKER_SHA = "eebf26505fbae7f03ae37415ce774c988c773eab412243e3291d599574bb6064"
PRIOR_EXPECTED_SHA = "28bcac0f2545182dcba6feedb9d750be4c6036ef34e6bfe1a7aefa6d3398921c"
PRIOR_TERMINAL_BUDGET = 1795508551.0896368
PCI = tuple(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
MARKERS = {
    OLD: {"claim.json": "4f972f7056552800ef73e8b0b9d722d8ac28adb027e1b2a71da8e132b24d2d31",
          "refill-grant.json": "b8912adab8fff537aa72afd18e990c0ee119a8b88d2c30f70bbb8aebd6b4141d",
          "released.json": None},
    PREVIOUS: {"claim.json": "82745962f7b5d1538f162f78aabeb3624625c6028743e2cd6947ca7e09984a84",
               "refill-grant.json": "a8da3e524a5642d744f8d2382cdc18d942f503d706a6b12787c2a7a82d1c4e28",
               "released.json": "b00fc3b0b523abd1dd9ac7106d7ad62ffb84867f99647ce019a0cad8c61ce5f9"},
}
CONTROLLER = "be504e7670b40d77e503c60c93d76bf2a75003768d764d54884c5f8272976bba"
SHA = re.compile(r"[a-f0-9]{64}")
ACCOUNTING = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded(path):
    with path.open("rb") as stream:
        raw = stream.read(1048577)
    require(0 < len(raw) <= 1048576, "bounded successor input")
    return raw


def digest(raw):
    return sha256(raw).hexdigest()


def directory(results):
    return results / (".training80-o01-successor-grant-" + REQUEST)


def assert_markers(results):
    original = results / (".training80-o01-successor-grant-" + PRIOR_REFILL)
    grant = original / "grant.json"
    require(not original.is_symlink() and not grant.is_symlink() and digest(bounded(grant)) == PRIOR_GRANT_SHA,
            "original successor grant differs")
    for request, pins in MARKERS.items():
        prior = results / (".training80-o01-pending-" + request)
        require(prior.is_dir() and not prior.is_symlink(), "original O01 reservation identity")
        for name, wanted in pins.items():
            path = prior / name
            if wanted is None:
                require(not os.path.lexists(path), "original reserved state changed")
            else:
                require(not path.is_symlink() and digest(bounded(path)) == wanted, "original reservation bytes changed")


def admit(inputs, certificate):
    proposal = {"parent": PARENT, "pci": list(PCI)}
    require(inputs["request"] == REQUEST and inputs["seconds"] == 10800
            and inputs["nodes"] == {NODE: proposal}, "exact one-node O01 proposal required")
    require(certificate["request_id"] == ADMISSION and certificate["payload_SHA256"] == inputs["admission_SHA256"]
            and SHA.fullmatch(certificate["payload_SHA256"])
            and re.fullmatch(r"1103606\.[0-9]+", certificate["device_probe_step"])
            and certificate["node"] == NODE and certificate["parent"] == PARENT
            and certificate["PCIs"] == list(PCI)
            and certificate["KFD_clean"] is True and certificate["PID_clean"] is True
            and certificate["kfd_process_count"] == certificate["own_device_fd_process_count"] == 0
            and certificate["active_steps"] == [] and certificate["device_probe_observed"] is True
            and certificate["ports_clean"] is True and certificate["membership_valid"] is True
            and certificate["probe_error"] is None, "fresh zero-GRES device certificate differs")


def accounting(step):
    cmd = ["sacct", "-n", "-P", "-j", step, "-o", ACCOUNTING]
    end = time.monotonic() + 6
    while True:
        out = subprocess.check_output(cmd, text=True, timeout=4)
        rows = [row for line in out.splitlines() if (row := line.split("|"))[0] == step]
        if len(rows) == 1 and len(rows[0]) == 9 and rows[0][2:4] == ["COMPLETED", "0:0"]:
            return rows[0]
        require(not rows or len(rows) == 1 and rows[0][2] == "RUNNING", "exact predecessor or probe did not complete")
        require(time.monotonic() < end, "exact predecessor or probe accounting not visible")
        time.sleep(0.35)


def require_prior_worker(results):
    folder = results / ("training80-dcu-standby-" + PRIOR_REFILL) / NODE
    expected, worker, terminal = (bounded(folder / name) for name in ("expected.json", "worker.json", "terminal.json"))
    require((digest(expected), digest(worker)) == (PRIOR_EXPECTED_SHA, PRIOR_WORKER_SHA),
            "prior worker or expected source differs")
    e, w, t = (json.loads(raw) for raw in (expected, worker, terminal))
    require(e["seconds"] == 10541 and e["request"] == PRIOR_REFILL and e["pci"] == list(PCI)
            and w["node"] == NODE and w["step"] == "1103606.2423" and w["worker_pid"] == 2093833
            and w["epoch"] == 1795498010.0896368 and w["pci"] == list(PCI)
            and t["node"] == NODE and t["step"] == "1103606.2423" and t["state"] == "WORKER_RETURNED"
            and PRIOR_TERMINAL_BUDGET - 1 <= t["epoch"] <= time.time() + 2,
            "prior device worker has not reached normal terminal")
    return digest(terminal)


def active_steps():
    numbered = re.compile(PARENT + r"\.[0-9]+")
    internal = {PARENT + ".batch", PARENT + ".extern"}
    host_lists = {}

    def includes_target(step, specs):
        if step in internal:
            return False
        require(numbered.fullmatch(step) and specs and specs not in {"(null)", "None", "Unknown", "N/A"},
                "unresolved numbered step membership")
        if specs not in host_lists:
            if re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", specs):
                expanded = [specs]
            else:
                expanded = subprocess.check_output(["scontrol", "show", "hostnames", specs], text=True,
                                                   timeout=5).splitlines()
            require(expanded and len(expanded) == len(set(expanded)) and all(
                re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", node) for node in expanded),
                "unknown expanded Slurm node membership")
            host_lists[specs] = frozenset(expanded)
        return NODE in host_lists[specs]

    control = ["scontrol", "show", "step", PARENT, "--oneliner"]
    steps = set()
    for line in subprocess.check_output(control, text=True, timeout=8).splitlines():
        if not line.strip():
            continue
        row = dict(token.split("=", 1) for token in line.split() if "=" in token)
        require(row.get("StepId"), "controller step identity missing")
        if includes_target(row["StepId"], row.get("NodeList")):
            steps.add(row["StepId"])
    queue = ["squeue", "--steps", "-h", "-j", PARENT, "-o", "%i|%N"]
    for line in subprocess.check_output(queue, text=True, timeout=8).splitlines():
        if not line.strip():
            continue
        row = line.split("|")
        require(len(row) == 2, "queue step columns changed")
        if includes_target(*row):
            steps.add(row[0])
    return sorted(steps)


def append(results, raw):
    grant_dir = directory(results)
    grant_dir.mkdir(mode=0o700)
    path = grant_dir / "grant.json"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    for folder in (grant_dir, results):
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def launch(results, envelope, entry, write_bridge):
    source_line, _, controller = entry.partition(b"\n")
    require(source_line.startswith(b"INPUT=") and digest(controller) == CONTROLLER
            and digest(entry) == envelope["entry_SHA256"], "frozen controller differs")
    proposal = ast.literal_eval(source_line[6:].decode("ascii"))
    certificate = envelope["o01_successor_device"]
    admit(proposal, certificate)
    require(envelope["request"] == REQUEST and 0 <= time.time() - certificate["observed_epoch"] <= 40,
            "successor request or authenticated certificate time differs")
    with (results / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_markers(results)
        require(not any(os.path.lexists(results / (prefix + REQUEST)) for prefix in
                        ("training80-dcu-standby-", "training80-standby-compute-")),
                "fixed successor already has an execution namespace")
        standby, sidecar, probe = (accounting(step) for step in
                                  ("1103606.2423", "1103606.2424", certificate["device_probe_step"]))
        require(standby[1] == "training80-dcu-standby-3f96a7be936a-f12r2n11"
                and sidecar[1] == "O01-owned-procfs-topology-2fe5248f9841"
                and "gres/dcu=8" in standby[6].split(",")
                and all(row[5] == NODE and row[8] != "Unknown" for row in (standby, sidecar, probe))
                and all(not any(key.startswith("gres/") for key in row[6].split(",")) for row in (sidecar, probe))
                and not active_steps() and 0 <= time.time() - certificate["observed_epoch"] <= 40,
                "predecessor, zero-GRES activity or live O01 occupancy differs")
        record = {"request": REQUEST, "scope_node": NODE, "parent": PARENT, "certificate": certificate,
                  "entry_SHA256": digest(entry), "predecessor": standby, "older_observer": sidecar,
                  "device_accounting": probe, "old_marker": OLD, "prior_grant_SHA256": PRIOR_GRANT_SHA,
                  "prior_worker_terminal_SHA256": require_prior_worker(results), "ordinary_dispatch_unblocked": False,
                  "epoch": time.time(), "replay_allowed": False}
        append(results, (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
        return write_bridge()


def install(results, namespace, inputs, entry):
    inventory = namespace["live_inventory"]

    def locked_inventory():
        owners, active = inventory()
        assert_markers(results)
        grant = json.loads(bounded(directory(results) / "grant.json"))
        require(grant["request"] == REQUEST and grant["scope_node"] == NODE
                and grant["parent"] == PARENT and grant["entry_SHA256"] == digest(entry)
                and grant["prior_grant_SHA256"] == PRIOR_GRANT_SHA and SHA.fullmatch(grant["prior_worker_terminal_SHA256"])
                and grant["ordinary_dispatch_unblocked"] is False and grant["replay_allowed"] is False
                and 0 <= time.time() - grant["epoch"] <= 90, "successor grant mismatch")
        admit(inputs, grant["certificate"])
        require(owners.get(NODE) == PARENT and NODE not in active and not active_steps(),
                "O01 no longer clear inside dispatch lock")
        return owners, active

    namespace["live_inventory"] = locked_inventory
