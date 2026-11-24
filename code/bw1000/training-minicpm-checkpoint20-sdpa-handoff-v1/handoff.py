"""Transfer the exact owned standby to the already-installed MiniCPM diagnostic."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REQUEST = "cbbd25487dca4090a52c923bbfb7f60b"
PARENT, NODE = "1103606", "f13r1n19"
STANDBY_STEP, STANDBY_REQUEST = "1103606.2208", "2e5a9fcb35134860a93091403b06f8a3"
STANDBY_NAME = "training80-dcu-standby-2e5a9fcb3513-f13r1n19"
STANDBY_TRES = "cpu=8,gres/dcu=8,mem=16G,node=1"
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
TARGET_NAME = "training-minicpm-checkpoint20-sdpa-repeatability-target-v1"
SOURCE = RESULTS / "training-minicpm-checkpoint20-sdpa-repeatability-source-cbbd25487dca"
TARGET = SOURCE / "code/bw1000" / TARGET_NAME
MARKER = RESULTS / ("training-minicpm-sdpa-handoff-" + REQUEST)
STANDBY = RESULTS / ("training80-dcu-standby-" + STANDBY_REQUEST) / NODE
TARGET_PUBLICATION = "2c6784036e865dbbe40adff4264c3221bb8462e2"
TARGET_SHA = "5bf1503b7af5863de7b3c8b5dc330d0e6f5bc1a8d0a20abab7988bf0cf8792b2"
STANDBY_EXPECTED_SHA = "7bdf34de2195fd9d8f7bdb279f0bc6fa1a7ddcf426f1890f61d3f52ad514a02e"
PYTHON = "/usr/local/python3.12/bin/python3"
RELEASE_SECONDS = 85


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path, expected=None, cap=8192):
    require(path.resolve(strict=True) == path, "aliased evidence path")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap, "unexpected evidence extent")
    if expected:
        require(sha256(raw).hexdigest() == expected, "exact source or evidence differs")
    return json.loads(raw)


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def verify_candidate(observation, expected, worker):
    parent = observation["parent"]
    require(parent["state"] == "RUNNING" and parent["nodes"] == 40 and parent["allocated_DCU"] == 320
            and all(parent[name] is True for name in ("ledger_exact", "step_locations_exact", "owned_by_observer"))
            and observation["original_diagnostic_unstarted"], "candidate parent or namespace differs")
    require(expected["request"] == STANDBY_REQUEST and expected["parent"] == PARENT
            and expected["node"] == worker["node"] == NODE and expected["label"] == STANDBY_NAME
            and worker["step"] == STANDBY_STEP and worker["synthetic"] is True
            and type(worker["worker_pid"]) is int and worker["worker_pid"] > 1
            and len(expected["pci"]) == len(set(expected["pci"])) == 8
            and sorted(worker["pci"]) == sorted(expected["pci"]), "predecessor identity differs")
    row = observation["accounting"]
    require(len(row) == 7 and row[:2] == [STANDBY_STEP, STANDBY_NAME]
            and row[5:] == [NODE, STANDBY_TRES], "predecessor accounting differs")
    active = observation["active_node_steps"]
    if active:
        require(set(active) == {STANDBY_STEP} and active[STANDBY_STEP]["member_nodes"] == [NODE]
                and active[STANDBY_STEP]["allocated_TRES"] == STANDBY_TRES,
                "another step occupies the intended node")
        if active[STANDBY_STEP]["state"] != "RUNNING":
            require(active[STANDBY_STEP]["state"] in {"COMPLETING", "COMPLETED", "CANCELLED"},
                    "unexpected controller standby state")
            return False
        require(row[2:4] == ["RUNNING", "0:0"], "standby accounting is inconsistent")
        return True
    require(row[2] in {"RUNNING", "COMPLETING", "COMPLETED", "CANCELLED"}, "unexpected predecessor state")
    return False


def released(observation, terminal):
    row = observation["accounting"]
    return (not observation["active_node_steps"] and row[:2] == [STANDBY_STEP, STANDBY_NAME]
            and row[2] in {"COMPLETED", "CANCELLED"} and row[3] == "0:0"
            and row[5:] == [NODE, STANDBY_TRES]
            and terminal is not None and terminal["node"] == NODE and terminal["step"] == STANDBY_STEP
            and terminal["state"] == "WORKER_RETURNED")


def bounded_transition(observe, terminal, verified, signal, start, note, *,
                       monotonic=time.monotonic, sleep=time.sleep):
    first = observe()
    active = verified(first)
    if active:
        require(terminal() is None, "running predecessor has a terminal marker")
        note("before-signal.json", first["accounting"])
        rc = signal()
        note("signal-return.json", {"rc": rc})
        # Nonzero can race a normal exit. Original accounting and worker proof still decide.
    deadline = monotonic() + RELEASE_SECONDS
    while monotonic() < deadline:
        proof = observe()
        verified(proof)
        require(proof["original_diagnostic_unstarted"], "diagnostic appeared; reconcile its original request")
        require(not set(proof["active_node_steps"]) - {STANDBY_STEP}, "another step entered during handoff")
        if released(proof, terminal()):
            note("standby-released.json", proof["accounting"])
            return start()
        sleep(2)
    raise TimeoutError("same predecessor still held or terminal proof incomplete")


def controller():
    manifest = read(TARGET / "SOURCE_FILES.json", TARGET_SHA)
    for name, pin in manifest["files"].items():
        require(Path(name).name == name, "unexpected source name")
        raw = (TARGET / name).read_bytes()
        require(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"], "installed source differs")
    sys.path.insert(0, str(TARGET))
    import current_admission as current
    import current_membership as membership
    import host_once as host
    previous, _ = host.verified(host.REPO)
    previous.controller(host.REPO)
    expected = read(STANDBY / "expected.json", STANDBY_EXPECTED_SHA)
    worker = read(STANDBY / "worker.json")
    ledger = read(TARGET / "current_ledger.json", current.LEDGER_SHA, 64 * 1024)
    groups = current.plan(ledger)
    owned_nodes = set().union(*groups[PARENT].values())

    def query(*argv):
        if argv[:3] == ("scontrol", "show", "hostnames") and argv[3] in owned_nodes:
            # A single literal from the exact ledger does not require range expansion.
            return [argv[3]]
        return membership.query(*argv)

    def observe():
        parent = membership.snapshot(groups, owner_uid=os.getuid(), run_query=query)["parents"][PARENT]
        steps = {step: row for step, row in parent["active_steps"].items()
                 if NODE in row["member_nodes"]}
        columns = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
        raw = subprocess.check_output(["sacct", "-n", "-P", "-j", PARENT,
                                       "-S", "2026-11-24T00:00:00", "-o", columns], text=True, timeout=8)
        rows = [line.split("|") for line in raw.splitlines() if line.strip()]
        require(all(len(row) == 7 for row in rows), "unexpected accounting columns")
        original = [row for row in rows if row[0] == STANDBY_STEP or row[1] == STANDBY_NAME]
        require(len(original) == 1, "original standby accounting is missing or duplicated")
        new_labels = {TARGET_NAME + "-" + phase for phase in host.PHASES}
        unused = not (RESULTS / TARGET_NAME).exists() and not any(row[1] in new_labels for row in rows)
        return dict(parent=parent, original_diagnostic_unstarted=unused,
                    accounting=original[0], active_node_steps=steps)

    def terminal():
        path = STANDBY / "terminal.json"
        return read(path) if path.exists() else None

    def launch(approval, pin):
        current.check(os.getuid())
        check = [PYTHON, "-B", "-X", "utf8", str(TARGET / "host_once.py")]
        bindings = ["--admission", str(approval), "--admission-sha256", pin, "--source-sha256", TARGET_SHA]
        with (MARKER / "preflight.log").open("xb") as stream:
            subprocess.run([*check, "--preflight", *bindings], stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=42, check=True)
        save(MARKER / "before-controller.json", {"request": REQUEST, "epoch": time.time(), "admission_SHA256": pin})
        with (MARKER / "target.log").open("xb") as stream:
            pid = subprocess.Popen([*check, "--launch-once", *bindings], stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        fields = Path(f"/proc/{pid.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        owner = {"pid": pid.pid, "pgid": int(fields[2]), "startticks": int(fields[19]),
                 "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        require(owner["pid"] == owner["pgid"], "actual controller process group differs")
        save(MARKER / "target-controller.json", dict(owner, request=REQUEST, admission_SHA256=pin))
        return owner

    return observe, terminal, lambda row: verify_candidate(row, expected, worker), launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-publication", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.handoff_publication), "handoff publication required")
    observe, terminal, verify, launch = controller()
    require(not MARKER.exists(), "original handoff marker already exists")
    command = [PYTHON, "-B", "-X", "utf8", str(TARGET / "host_once.py")]
    subprocess.run([*command, "--asset-preflight", "--source-sha256", TARGET_SHA],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=42, check=True)
    os.umask(0o077)
    MARKER.mkdir(mode=0o700)
    save(MARKER / "handoff-intent.json",
         {"request": REQUEST, "standby": STANDBY_STEP, "node": NODE, "parent": PARENT,
          "handoff_publication": args.handoff_publication, "target_publication": TARGET_PUBLICATION,
          "target_manifest_SHA256": TARGET_SHA, "replay_allowed": False, "epoch": time.time()})
    approval = MARKER / "admission.json"
    adopted = read(TARGET / "CLAIM.json")
    adopted.update(publication=TARGET_PUBLICATION, root_GO=True, GPU_GO=True)
    pin = save(approval, adopted)

    def signal():
        return subprocess.run(["scancel", "--signal=TERM", STANDBY_STEP], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8).returncode

    try:
        owner = bounded_transition(observe, terminal, verify, signal, lambda: launch(approval, pin),
                                   lambda name, data: save(MARKER / name, data))
        print(json.dumps({"request": REQUEST, "controller": owner, "admission_SHA256": pin,
                          "slurm_numbered_submit_proven_here": False}, sort_keys=True))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        save(MARKER / "handoff-error.json", {"request": REQUEST, "failure_class": type(error).__name__,
                                             "reason": str(error)[:240], "replay_allowed": False})
        raise


if __name__ == "__main__":
    main()
