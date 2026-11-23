"""Extend the qualified colocated BF16 trial with the same guard and a finite longer budget."""

from hashlib import sha256
import json
import re
import subprocess
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(root, claim, worker, controller, pilot):
    import fcntl

    require(sha256(pilot).hexdigest() == claim["pilot_module_SHA256"], "pilot guard digest drift")
    source = {"__name__": "qualified_shared_standby_guard"}
    exec(compile(pilot, "qualified-pilot-guard.py", "exec"), source)
    guard, query, rows, probe, save = [source[name] for name in ("guard", "query", "rows", "probe", "save")]
    PARENT, NODE, PROBE, ORIGINAL_PROBE, ACCOUNTING_COLUMNS = [
        source[name] for name in ("PARENT", "NODE", "PROBE", "ORIGINAL_PROBE", "ACCOUNTING_COLUMNS")]

    request = root.name.removeprefix("training80-dcu-standby-")
    require(re.fullmatch(r"[a-f0-9]{32}", request), "invalid pilot request identity")
    require(sha256(worker).hexdigest() == claim["worker_SHA256"]
            and sha256(controller).hexdigest() == claim["controller_SHA256"]
            and all(sha256((PROBE / file).read_bytes()).hexdigest() == pin
                    for file, pin in claim["probe_source_SHA256"].items()), "pilot dependency identity drift")
    namespace = {"__name__": "standby_v2_qualified_controller"}
    exec(compile(controller, "qualified-controller.py", "exec"), namespace)
    require(any(row[2:4] == ["COMPLETED", "0:0"] and row[5] == NODE and row[6] == "cpu=1,mem=256M,node=1"
                for row in rows(ORIGINAL_PROBE)), "original zero-GRES probe has no matching terminal accounting")
    label = f"training80-dcu-standby-{request[:12]}-{NODE}"
    require(not any(row[1] == label for row in [line.split("|") for line in query(
        "sacct", "-n", "-P", "-j", PARENT, "-o", ACCOUNTING_COLUMNS)] if len(row) == 7), "pilot label already used")
    previous = claim["qualified_pilot"]
    pilot_label = f'training80-dcu-standby-{previous["request"][:12]}-{NODE}'
    [terminal] = rows(previous["step"])
    require(terminal[1:4] == [pilot_label, "COMPLETED", "0:0"]
            and terminal[5:] == [NODE, "cpu=8,gres/dcu=8,mem=16G,node=1"], "pilot lacks exact successful terminal accounting")
    previous_directory = root.with_name("training80-dcu-standby-" + previous["request"]) / NODE
    worker_terminal = json.loads((previous_directory / "terminal.json").read_bytes())
    require(worker_terminal["node"] == NODE and worker_terminal["step"] == previous["step"]
            and worker_terminal["state"] == "WORKER_RETURNED", "pilot did not return normally")
    seconds = claim["worker_seconds"]
    require(seconds == 10500 and claim["slurm_seconds"] == seconds + 300, "invalid standby budget")
    with (root.parent / ".training80-dcu-standby.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(claim, namespace)
        folder = root / NODE
        folder.mkdir(parents=False)
        pre = probe(request, claim)
        guard(claim, namespace)
        save(folder / "expected.json", dict(node=NODE, parent=PARENT, pci=claim["pci"], seconds=seconds,
                                           label=label, request=request))
        save(root / "intent.json", dict(request=request, node=NODE, parent=PARENT, pre=pre,
                                        epoch=time.time(), synthetic=True, replay_allowed=False))
        argv = namespace["command"](NODE, PARENT, label, root, folder, seconds)
        with (folder / "slurm.stdout.private").open("xb") as out, (folder / "slurm.stderr.private").open("xb") as err:
            process = subprocess.Popen([argv[0], "--threads=1", *argv[1:]], stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True)
        save(root / "submit.json", dict(request=request, label=label, launcher_pid=process.pid,
                                        worker_seconds=seconds, synthetic=True))
    save(root / "submission-receipt.json", dict(request=request, node=NODE, parent=PARENT, synthetic=True,
          submit_records=[dict(node=NODE, parent=PARENT, label=label, launcher_pid=process.pid, worker_seconds=seconds)]))
    print(json.dumps(dict(request=request, label=label, node=NODE, pre=pre, launcher_pid=process.pid)), flush=True)
