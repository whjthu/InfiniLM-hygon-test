"""Check one frozen standby batch, then pass exact released nodes to the existing refill."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, time
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
PLAN = LOCAL / "training80-1738-ordinary-renewal-batches-v1.json"
PLAN_SHA = "ccff7a742a301353fb24a88ce3eba477290995b0b08a68802c346bd752dc8868"
READER = LOCAL / "read-training80-1738-ordinary-exact-batch-terminal-v1.py"
READER_SHA = "7aa863d63ff13430ec9e3a8c1c5ebfb9ff001a5147d091d3c807259c1b0cb5f4"
REFILL = REPO.parent / "run-dcu-standby-cycle-20261123.ps1"
REFILL_SHA = "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73"
BOUND_SECONDS = 35
ADMISSION_FIELDS = {"admission", "current_running_numbered_DCU", "all_reported_numbered_DCU_lower_bound"}


def pinned(path, expected):
    if path.is_symlink():
        raise ValueError(f"unexpected link: {path.name}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected bytes: {path.name}")
    return raw


def released_nodes(rows, receipt):
    observed = receipt["observed"]
    if receipt["rc"] != 0 or observed is None:
        return []
    if observed["plans_only"] is not True or observed["requires_fresh_independent_device_and_dispatch_checks"] is not True:
        raise ValueError("wrong terminal receipt")
    targets = observed["targets"]
    if len(targets) != len(rows):
        raise ValueError("terminal receipt omitted planned nodes")
    released = []
    for row, target in zip(rows, targets, strict=True):
        if any(target[a] != row[b] for a, b in (("step", "previous"), ("node", "node"), ("request", "request"))):
            raise ValueError("terminal receipt crossed predecessor scope")
        if target["terminal_ready"] is not True:
            continue
        accounting = target["accounting"]
        if (not isinstance(accounting, list) or len(accounting) != 9
                or accounting[0] != row["previous"] or accounting[2:4] != ["COMPLETED", "0:0"]
                or accounting[5:7] != [row["node"], "cpu=8,gres/dcu=8,mem=16G,node=1"]
                or not accounting[8] or accounting[8] == "Unknown"
                or target["worker_SHA256"] != row["worker_SHA256"] or target["expected_SHA256"] != row["expected_SHA256"]
                or not re.fullmatch("[0-9a-f]{64}", target["terminal_SHA256"])):
            raise ValueError(f"terminal evidence does not qualify: {row['node']}")
        released.append(row)
    return released


def store(path, data):
    with path.open("xb") as stream:
        stream.write((json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def reserve_node(journal, node, step, intent):
    for attempt in range(3):
        path = journal / f"{node}-{step}-{attempt}.json"
        try:
            store(path, intent)
        except FileExistsError:
            prior = json.loads(path.read_bytes())["wrapper_invocation"]
            if not re.fullmatch("[0-9a-f]{32}", prior):
                raise ValueError("invalid previous refill reservation")
            outcome = journal / f"{prior}-outcome.json"
            if not outcome.is_file():
                return False
            previous = json.loads(outcome.read_bytes())
            if (previous["wrapper_invocation"] != prior or node not in previous["terminal_ready_nodes"]
                    or previous["no_GPU_send_confirmed"] is not True):
                return False
        else:
            return True
    return False


def did_not_send(stdout, rc):
    if rc != 0 or len(stdout) > 65536:
        return False
    try:
        lines = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except (UnicodeError, ValueError):
        return False
    if not lines or not all(isinstance(value, dict) for value in lines):
        return False
    final = lines[-1]
    if set(final) == {"standby_no_dispatch"} and isinstance(final["standby_no_dispatch"], str):
        return len(lines) == 1 or (len(lines) == 2 and set(lines[0]) == ADMISSION_FIELDS)
    return (len(lines) == 1 and set(final) == ADMISSION_FIELDS and isinstance(final["admission"], dict)
            and type(final["admission"].get("clear_cards")) is int and final["admission"]["clear_cards"] == 0)


def run(batch, allow_dispatch):
    batches = json.loads(pinned(PLAN, PLAN_SHA))["candidate_batches_in_chronological_budget_order"]
    if not 1 <= batch <= len(batches):
        raise ValueError("batch lies outside the frozen plan")
    rows = batches[batch - 1]["rows"]
    if not 1 <= len(rows) <= 8:
        raise ValueError("batch exceeds the verified refill size")
    pinned(READER, READER_SHA)
    pinned(REFILL, REFILL_SHA)
    if not allow_dispatch:
        return {"batch": batch, "nodes": [row["node"] for row in rows], "remote_calls": 0}

    started = monotonic()
    done = subprocess.run([sys.executable, "-B", "-X", "utf8", str(READER), "--batch", str(batch)],
                          cwd=REPO, capture_output=True, timeout=BOUND_SECONDS, check=True)
    if not 0 < len(done.stdout) <= 8192:
        raise ValueError("unexpected terminal reader output size")
    hint = json.loads(done.stdout)
    name = hint["file"]
    if not re.fullmatch(rf"training80-1738-ordinary-terminal-batch-{batch}-[0-9a-f]{{32}}\.json", name):
        raise ValueError("terminal reader returned an unexpected path")
    observed = json.loads(pinned(LOCAL / name, hint["SHA256"]))
    released = released_nodes(rows, observed)
    if hint["ready_only_for_device_probe"] != [row["node"] for row in released]:
        raise ValueError("terminal reader node list differs from the original")
    if not released:
        return {"batch": batch, "terminal_receipt": name, "ready_for_device_probe": [],
                "refill_attempted": False, "observation_rc": observed["rc"]}

    if monotonic() - started >= BOUND_SECONDS:
        raise TimeoutError("terminal reader receipt has aged before refill")
    journal = LOCAL / "training80-1738-ordinary-renewal-dispatch"
    journal.mkdir(exist_ok=True)
    request = uuid4().hex
    selected = []
    for row in released:
        node, step = row["node"], row["previous"]
        intent = {"batch": batch, "node": node, "predecessor": step, "terminal_receipt": name,
                  "terminal_receipt_SHA256": hint["SHA256"], "wrapper_invocation": request,
                  "replay_allowed": False, "recorded_epoch": time()}
        if reserve_node(journal, node, step, intent):
            selected.append(node)
    if not selected:
        return {"batch": batch, "terminal_receipt": name, "refill_attempted": False,
                "nodes_with_existing_attempt_marks": [row["node"] for row in released]}

    env = {**os.environ, "HYGON_STANDBY_ONLY_NODES": " ".join(selected), "HYGON_STANDBY_VIA_COMPUTE": "1"}
    result = {"batch": batch, "wrapper_invocation": request, "terminal_receipt": name,
              "terminal_ready_nodes": selected, "refill_attempted": True}
    try:
        run = subprocess.run(["pwsh", "-NoProfile", "-File", str(REFILL)], cwd=REPO,
                             env=env, capture_output=True, timeout=190, check=False)
    except subprocess.TimeoutExpired as error:
        result.update(refill_rc=None, transport_observation="TIMEOUT_ORIGINAL_SCOPE_REQUIRES_RECONCILIATION")
        stdout, stderr = error.stdout or b"", error.stderr or b""
    else:
        result.update(refill_rc=run.returncode, transport_observation="REFILL_PROCESS_RETURNED")
        stdout, stderr = run.stdout, run.stderr
    result["no_GPU_send_confirmed"] = did_not_send(stdout, result["refill_rc"])
    for stream, raw in (("stdout", stdout), ("stderr", stderr)):
        with (journal / f"{request}-{stream}.private").open("xb") as handle:
            handle.write(raw)
        result[stream + "_SHA256"] = sha256(raw).hexdigest()
    store(journal / f"{request}-outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--dispatch", action="store_true", help="read actual terminal accounting and try one refill")
    args = parser.parse_args()
    print(json.dumps(run(args.batch, args.dispatch), sort_keys=True))


if __name__ == "__main__":
    main()
