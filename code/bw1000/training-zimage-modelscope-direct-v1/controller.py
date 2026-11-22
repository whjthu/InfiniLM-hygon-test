"""Own one bounded Slurm data-transfer step inside the T07 allocation."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = "/usr/local/python3.12/bin/python3"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")


def require(value, message):
    if not value:
        raise ValueError(message)


def values(line):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line))


def save(path, report):
    raw = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def main():
    claim = json.loads((ROOT/"CLAIM.json").read_bytes())
    intent = json.loads((ROOT/"intent.json").read_bytes())
    epoch = intent["epoch"]
    parent, node, label = (claim[key] for key in ("parent", "node", "label"))
    require(intent["request"] == claim["request"] and label == ROOT.name and
            epoch <= time.time() < epoch+12, "Original startup identity differs")
    report = dict(request=claim["request"], accounting_columns=COLUMNS, launch_attempted=False,
                  state="PRELAUNCH", epoch=epoch, accounting=None)

    def query(*args):
        return subprocess.check_output(args, timeout=10, text=True)

    def row():
        raw = query("sacct", "-n", "-P", "-j", parent, "--format="+",".join(COLUMNS))
        matches = [r.split("|") for r in raw.splitlines() if r.split("|")[1] == label]
        require(len(matches) <= 1, "Ambiguous direct Slurm identity")
        return matches[0] if matches else None

    def wait_terminal():
        cutoff = min(epoch+claim["whole_seconds"]-2, time.time()+22)
        while time.time() < cutoff:
            observed = row()
            if observed and observed[2] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                                             "OUT_OF_MEMORY", "PREEMPTED", "DEADLINE", "BOOT_FAIL"}:
                return observed
            time.sleep(.5)
        return None

    try:
        flags = values(query("scontrol", "show", "job", parent, "-o"))
        require(flags["JobState"] == "RUNNING" and flags["UserId"].endswith(f"({os.getuid()})")
                and flags["NumNodes"] == "40" and "gres/dcu=320" in flags["AllocTRES"].split(","),
                "Owned T07 parent identity differs")
        require(node in query("scontrol", "show", "hostnames", flags["NodeList"]).splitlines(),
                "Exact T07 node is unavailable")
        for line in query("scontrol", "show", "step", parent, "-o").splitlines():
            step = values(line)
            if step["StepId"] in (parent+".batch", parent+".extern"):
                continue
            require(re.fullmatch(parent+r"\.\d+", step["StepId"]) and step["NodeList"] not in ("", "(null)")
                    and node not in query("scontrol", "show", "hostnames", step["NodeList"]).splitlines(),
                    "An existing step owns the target node")
        manifest = (ROOT/"SOURCE_FILES.json").read_bytes()
        require(sha256(manifest).hexdigest() == intent["manifest_SHA256"], "Published source manifest differs")
        for name, pin in json.loads(manifest)["files"].items():
            raw = (ROOT/name).read_bytes()
            require(Path(name).name == name and len(raw) == pin["bytes"]
                    and sha256(raw).hexdigest() == pin["SHA256"], "Published member differs")
        require(row() is None, "Original direct label already exists")
        budget = epoch+claim["worker_seconds"]+12-time.time()
        require(budget > 90, "Insufficient original worker budget")
        args = ["srun", "--jobid="+parent, "--job-name="+label, "--nodelist="+node,
                "--nodes=1", "--ntasks=1", "--exact", "--cpus-per-task=2", "--mem=4096M",
                "--gres=none", "--immediate=5", "--time="+str(int(budget//60)+1),
                PYTHON, "-I", "-S", "-u", "-B", str(ROOT/"direct.py"),
                "--epoch", repr(epoch), "--request", claim["request"]]
        with (ROOT/"srun.log").open("xb") as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report["launch_attempted"] = True
            try:
                report["child_rc"] = child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    report["child_rc"] = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    report["child_rc"] = child.wait(timeout=3)
        actual = wait_terminal()
        report["accounting"] = actual
        require(actual is not None and actual[5] == node and "gres/dcu=" not in actual[6],
                "Original zero-GRES terminal evidence is incomplete")
        report["state"] = "ACTUAL_TERMINAL"
    except Exception as error:
        report.update(error_type=type(error).__name__, brief=str(error)[:180])
        if report["launch_attempted"]:
            report["state"] = "ORIGINAL_TERMINAL_UNCONFIRMED"
    save(ROOT/"outcome.json", report)


if __name__ == "__main__":
    main()
