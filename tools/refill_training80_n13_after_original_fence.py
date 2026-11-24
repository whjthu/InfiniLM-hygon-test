"""One independently guarded low-priority BF16 refill of f12r2n13."""

from hashlib import sha256
import json
import os
import re
import subprocess
import time


OLD_REQUEST = "b4c58b1681d5485e91d2c76ac15aad02"
NODE = "f12r2n13"
PARENT = "1103606"
FENCE = "1103606.2311"
FENCE_LABEL = "training80-standby-compute-ff3e25d8d209"
CONTROL_TRES = "cpu=2,mem=2G,node=1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bind_controls(rows, allowed, optional_label):
    controls = {}
    for step, (label, state, resources) in rows.items():
        if step == "1103606.1691":
            require((label, state, resources) == ("training80-syncbusy-5af4008d8703-control", "TIMEOUT", "cpu=2,mem=512M,node=1"),
                    "protected zero-GRES step changed")
            continue
        require(state == "RUNNING" and resources == CONTROL_TRES and
                (allowed.get(step) == label or label == optional_label) and label not in controls.values(),
                "a control is outside the bound zero-GRES scope")
        controls[step] = label
    return controls


def source(raw, name):
    module = {"__name__": "pinned_" + name}
    exec(compile(raw, name + ".py", "exec"), module)
    return module


def run(claim, files, publication):
    import fcntl

    require(set(files) == {"pilot", "controller", "worker"} and
            {name: sha256(raw).hexdigest() for name, raw in files.items()} == claim["dependency_SHA256"], "source pin differs")
    request = claim["request"]
    require(re.fullmatch(r"[a-f0-9]{40}", publication) and re.fullmatch(r"[a-f0-9]{32}", request)
            and request != OLD_REQUEST and claim["parent"] == PARENT and claim["node"] == NODE
            and (claim["GPU_cards"], claim["worker_seconds"], claim["slurm_seconds"]) == (8, 10500, 10800)
            and claim["old_request"] == OLD_REQUEST and claim["replay"] is False and claim["synthetic"] is True,
            "publication, identity or budget differs")
    pilot = source(files["pilot"], "pilot")
    controller = source(files["controller"], "controller")
    results = controller["RESULTS"]
    query = pilot["query"]
    original = results / ("training80-dcu-standby-" + OLD_REQUEST)
    root = results / ("training80-dcu-standby-" + request)
    label = "training80-dcu-standby-" + request[:12] + "-" + NODE
    probe_label = "training80-admit-control-pilot-" + request
    old_labels = {"training80-dcu-standby-" + OLD_REQUEST[:12] + "-" + NODE,
                  "training80-admit-control-pilot-" + OLD_REQUEST}

    def verify_dependencies():
        pins = claim["probe_source_SHA256"]
        for name, digest in pins.items():
            require(sha256((pilot["PROBE"] / name).read_bytes()).hexdigest() == digest, "zero-GRES probe dependency drift")

    def admission_source(check_new_probe):
        try:
            original.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("original successor namespace now exists")
        [fence] = pilot["rows"](FENCE)
        require(fence[:4] == [FENCE, FENCE_LABEL, "COMPLETED", "0:0"] and fence[5:] == [NODE, CONTROL_TRES]
                and claim["bound_ended_original_control"] == FENCE, "original control fence is not terminal")
        [previous] = pilot["rows"]("1103606.2301")
        require(previous[:4] == ["1103606.2301", "training80-dcu-standby-d97dfeabcf50-" + NODE, "COMPLETED", "0:0"]
                and previous[5:] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"], "previous standby terminal changed")
        forbidden = {*old_labels, label} if check_new_probe else {*old_labels, label, probe_label}
        accounting = query("sacct", "-nP", "-S", "2026-11-24T14:20:00", "-j", PARENT, "-o", pilot["ACCOUNTING_COLUMNS"])
        for raw in accounting:
            record = raw.split("|")
            require(len(record) == 7 and record[1] not in forbidden, "original or new request label is visible in accounting")
        local = {}
        host_lists = {}
        for line in query("scontrol", "show", "step", PARENT, "--oneliner"):
            row = controller["pinned_fields"](line)
            step = row["StepId"]
            if step in (PARENT + ".batch", PARENT + ".extern"):
                continue
            require(re.fullmatch(PARENT + r"\.[0-9]+", step) and row["Name"] not in forbidden, "conflicting live step")
            nl = row["NodeList"]
            if nl not in host_lists:
                host_lists[nl] = [nl] if re.fullmatch(r"f\d+r\d+n\d+", nl) else query("scontrol", "show", "hostnames", nl)
            if NODE in host_lists[nl]:
                require(host_lists[nl] == [NODE] and step not in local, "unbound shared node step")
                local[step] = (row["Name"], row["State"], pilot["step_resources"](row))
        controls = bind_controls(local, claim["observed_allowed_control_steps"], claim["allowed_T05_control_label"])
        require(len(controls) <= 23, "live control count exceeds frozen capacity budget")
        guarded = dict(claim, zero_GRES_controls=controls)
        pilot["guard"](guarded, controller)
        return guarded

    os.umask(0o077)
    with (results / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_dependencies()
        guarded = admission_source(False)
        root.mkdir(mode=0o700)
        node_dir = root / NODE
        node_dir.mkdir(mode=0o700)
        write = controller["write"]
        write(root, "preflight.json", {"request": request, "parent": PARENT, "node": NODE, "epoch": time.time(),
              "publication": publication, "ended_original_control": FENCE, "observed_controls": guarded["zero_GRES_controls"]})
        pre = pilot["probe"](request, guarded)
        deadline = time.monotonic() + 6
        while True:
            rows = pilot["rows"](pre["step"])
            if (len(rows) == 1 and rows[0][:4] == [pre["step"], probe_label, "COMPLETED", "0:0"]
                    and rows[0][5:] == [NODE, "cpu=1,mem=256M,node=1"]):
                break
            require(time.monotonic() < deadline, "fresh zero-GRES probe lacks terminal accounting")
            time.sleep(0.4)
        guarded = admission_source(True)
        with (root / "worker.py").open("xb") as file:
            file.write(files["worker"])
        seconds = claim["worker_seconds"]
        write(node_dir, "expected.json", {"request": request, "parent": PARENT, "node": NODE,
              "label": label, "pci": claim["pci"], "seconds": seconds})
        write(root, "intent.json", {"request": request, "node": NODE, "parent": PARENT, "epoch": time.time(),
              "publication": publication, "synthetic": True, "pre": pre, "replay_allowed": False,
              "preserved_controls": guarded["zero_GRES_controls"]})
        argv = controller["command"](NODE, PARENT, label, root, node_dir, seconds)
        with (node_dir / "slurm.stdout.private").open("xb") as out, (node_dir / "slurm.stderr.private").open("xb") as err:
            process = subprocess.Popen([argv[0], "--threads=1", *argv[1:]], stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True)
        record = {"request": request, "node": NODE, "parent": PARENT, "label": label, "launcher_pid": process.pid,
                  "worker_seconds": seconds, "pre": pre, "preserved_controls": len(guarded["zero_GRES_controls"])}
        write(root, "submit.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)


def main():
    import argparse
    from base64 import b64encode
    from pathlib import Path
    import shlex
    import sys
    from types import ModuleType

    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = cli.parse_args()
    repo = Path(__file__).resolve().parent.parent
    local = repo / "local-only"
    claim_path = "coordination/tasks/training80-n13-standby-after-b4c58b-original-fence-v1.json"
    claim_raw = (repo / claim_path).read_bytes()
    claim = json.loads(claim_raw)
    request = claim["request"]
    script = Path(__file__).read_bytes()
    files = {name: (repo / path).read_bytes() for name, path in claim["dependency_path"].items()}
    require({name: sha256(raw).hexdigest() for name, raw in files.items()} == claim["dependency_SHA256"], "local dependency pin differs")
    receipt = local / claim["read_only_evidence"]["file"]
    raw = receipt.read_bytes()
    require(sha256(raw).hexdigest() == claim["read_only_evidence"]["sha256"], "original read-only evidence differs")
    evidence = json.loads(raw)
    observed = evidence["observed"]
    require(evidence["rc"] == 0 and observed["request"] == OLD_REQUEST and observed["node"] == NODE
            and observed["namespace_before"]["status"] == observed["namespace_after"]["status"] == "ABSENT"
            and not observed["controller_matches"] and not observed["accounting_matches"], "original request still unresolved")
    observed_rows = {step: tuple(row[:3]) for step, row in observed["controller_node_steps"].items() if row[3]}
    require(bind_controls(observed_rows, claim["observed_allowed_control_steps"], claim["allowed_T05_control_label"])
            == claim["observed_allowed_control_steps"], "original control evidence differs from claim")
    require(any(row[0:4] == [FENCE, FENCE_LABEL, "COMPLETED", "0:0"] and row[8] != "Unknown"
                for row in observed["accounting_old_and_required_controls"]), "no irreversible original control fence")

    sys.path.insert(0, str(repo / "tools"))
    import report_training80_backlog as backlog_reader
    holds = backlog_reader.read_queue_holds(local / "tuning/queue.sqlite3")
    backlog = json.loads((repo / "coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    nodes = set(backlog_reader.allocation(backlog_reader.TRAINING80_LEDGER.read_bytes())[2])
    for item in backlog:
        nodes.update(item["placement"])
    original_holds, _, _ = backlog_reader.read_original_holds(repo, backlog, nodes, holds)
    require(not any(NODE in row["nodes"] for row in [*holds, *original_holds.values()]) and
            not any(NODE in item["placement"] and item["kind"] == "training" and item["state"] == "ready" for item in backlog),
            "higher priority or unreconciled local task holds this node")
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    transport_source = helper.read_bytes()
    require(sha256(transport_source).hexdigest() == "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd",
            "existing strict transport differs")
    if args.check:
        print(json.dumps({"local_check": True, "request": request, "observed_controls": len(observed_rows) - 1, "remote_calls": 0}))
        return 0

    require(re.fullmatch(r"[a-f0-9]{40}", args.publication), "publication must be an exact commit")
    relative = Path(__file__).resolve().relative_to(repo).as_posix()
    for path, expected in ((relative, script), (claim_path, claim_raw)):
        actual = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "show", f"{args.publication}:{path}"], cwd=repo)
        require(actual == expected or actual == expected.replace(b"\r\n", b"\n"), "local source differs from publication")
    command_source = r'''
from base64 import b64decode
from hashlib import sha256
import json, sys
data = sys.stdin.buffer.read(131073)
assert 0 < len(data) <= 131072
packet = json.loads(data)
payloads = {key: b64decode(value, validate=True) for key, value in packet["payloads"].items()}
assert {key: sha256(raw).hexdigest() for key, raw in payloads.items()} == packet["sha256"]
namespace = {"__name__": "published_n13_refill"}
exec(compile(payloads.pop("script"), "published_n13_refill.py", "exec"), namespace)
claim = json.loads(payloads.pop("claim"))
try:
    namespace["run"](claim, payloads, packet["publication"])
except (KeyError, OSError, ValueError, RuntimeError) as error:
    print(json.dumps({"request": claim["request"], "guard_failure": type(error).__name__,
                      "reason_SHA256": sha256(str(error).encode()).hexdigest()}), flush=True)
    raise SystemExit(1)
'''
    compile(command_source, "<remote payload>", "exec")
    encoded = {"script": script, "claim": claim_raw, **files}
    packet = json.dumps({"payloads": {name: b64encode(raw).decode() for name, raw in encoded.items()},
                        "sha256": {name: sha256(raw).hexdigest() for name, raw in encoded.items()}, "publication": args.publication},
                       sort_keys=True).encode()
    helper_module = ModuleType("qualified_existing_transport")
    exec(compile(transport_source, str(helper), "exec"), vars(helper_module))
    command = [*helper_module.ssh(repo.parent), shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", command_source])]

    def save(stem, value):
        path = local / f"training80-n13-post-original-fence-{stem}-{request}.json"
        data = (json.dumps(value, sort_keys=True) + "\n").encode()
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        return path.name, sha256(data).hexdigest()

    save("send", {"request": request, "publication": args.publication, "packet_SHA256": sha256(packet).hexdigest(), "replay_allowed": False})
    try:
        result = subprocess.run(command, input=packet, capture_output=True, timeout=55, check=False)
        rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    try:
        row = json.loads(stdout) if 0 < len(stdout) <= 16384 else None
    except (UnicodeError, ValueError):
        row = None
    record = {"rc": rc, "request": request, "stdout_SHA256": sha256(stdout).hexdigest(),
              "stderr_SHA256": sha256(stderr).hexdigest(), "stderr_bytes": len(stderr), "row": row, "replay_allowed": False}
    file, digest = save("reply", record)
    print(json.dumps({"file": file, "sha256": digest, **record}, sort_keys=True))
    return int(rc != 0 or row is None)


if __name__ == "__main__":
    raise SystemExit(main())
