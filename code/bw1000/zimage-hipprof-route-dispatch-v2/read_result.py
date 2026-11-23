"""Read the original pair, actual accounting and independent cleanup receipts."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL = "O01-zimage-square-mm-hipprof-route-v2"
REQUEST = "9429f548ca3346c683f7361037079f75"
RUN = ROOT / (LABEL + "-run")
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "DEADLINE", "BOOT_FAIL"}


def document(name):
    path = RUN / name
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        raw = stream.read(262145)
    if len(raw) > 262144:
        raise ValueError("Bounded original metadata exceeded")
    return {"SHA256": sha256(raw).hexdigest(), "bytes": len(raw), "record": json.loads(raw)}


def observe():
    raw = subprocess.check_output(["sacct", "-nP", "-j", "1103606", "--format", ",".join(COLUMNS)],
                                  text=True, timeout=7)
    rows = [row.split("|") for row in raw.splitlines()
            if "|" + LABEL + "-" in row or row.startswith("1103606.1873|")]
    if not all(len(row) == len(COLUMNS) for row in rows):
        raise ValueError("Unexpected original accounting schema")
    outcome = document("outcome.json")
    control = document("control-owner.json")
    ended_control = bool(control and any(row[0] == control["record"]["step"]
                        and row[1] == LABEL + "-control" and row[2].rstrip("+") in TERMINAL
                        and row[5] == "f12r2n13" and "gres/" not in row[6] for row in rows))
    terminal = bool(outcome and ended_control and all(row[2].rstrip("+") in TERMINAL for row in rows))
    value = dict(observed_epoch=time.time(), request=REQUEST, namespace_exists=RUN.is_dir(), accounting=rows,
                 accounting_columns=COLUMNS, controller_terminal=terminal,
                 intent=document("intent.json"), control_owner=control,
                 gpu_intent=document("gpu-attempt.json"), outcome=outcome)
    report = document("worker-output/terminal.json")
    if not terminal:
        if report:
            r = report["record"]
            value["worker_progress"] = {k: r[k] for k in ("completed", "passed", "step", "layout_conclusion") if k in r}
        return value

    pre, post, scope = (document(name) for name in ("pre.json", "post.json", "worker-scope.json"))
    value["metadata_pins"] = {name: {k: row[k] for k in ("SHA256", "bytes")} for name, row in
                              (("independent_pre", pre), ("independent_post", post), ("worker_scope", scope))
                              if row}
    if pre and post:
        p, q = pre["record"], post["record"]
        clean = (len(set(p["PCIs"])) == 8 and set(p["PCIs"]) == set(q["PCIs"])
                 and p["boot_id_SHA256"] == q["boot_id_SHA256"]
                 and p["node"] == q["node"] == "f12r2n11"
                 and q["kfd_process_count"] == q["own_device_fd_process_count"] == 0
                 and any(row[0] == q["step"] and row[2:4] == ["COMPLETED", "0:0"]
                         and "gres/" not in row[6] for row in rows))
        if bool(outcome["record"]["post_clean"]) != clean:
            raise ValueError("Original outcome and independent clean check differ")
        value["independent_device_clear"] = clean
    else:
        value["independent_device_clear"] = False

    if report:
        r = report["record"]
        if (r["request"] != REQUEST or r["node"] != "f12r2n11" or r["planned_cells"] != 4
                or r["timed_calls"] != 0 or not scope
                or r["scope_SHA256"] != scope["SHA256"] or not any(row[0] == r["step"]
                    and row[5] == "f12r2n11" and "gres/dcu=1" in row[6].split(",") for row in rows)):
            raise ValueError("Actual worker and Slurm or measurement scope differ")
        value["worker"] = report
    return value


if __name__ == "__main__":
    print(json.dumps(observe(), sort_keys=True, allow_nan=False))
