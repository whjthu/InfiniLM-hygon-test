"""Reconcile one exact parent-local ordinary group and refill its released subset."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import sys
from time import monotonic, time
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
JOURNAL = LOCAL / "training80-1738-ordinary-renewal-dispatch"
CALENDAR = LOCAL / "training80-2347-ordinary-70-exact-predecessor-calendar-preparation-v1.json"
CALENDAR_SHA = "f70f6023e2469d624201e9c1fa28abf573e5d091214ac6c85bc45cafd997d6c8"
OLD_WRAPPER = LOCAL / "run-training80-2035-ordinary-renewal-once.py"
OLD_WRAPPER_SHA = "62156f03554141703c55746165e4be8f7e41f4abb7b192d079bc764eb963e720"
OLD_READER = LOCAL / "read-training80-2035-ordinary-exact-batch-terminal-v1.py"
OLD_READER_SHA = "608f3dada8c2bc2c036b1b0f12b59855b78fe3e4c251afa75a2692c565d27c98"
PUBLISHER = REPO / "tools/run_standby_renewal_with_pinned_publication.py"
PUBLISHER_SHA = "e900334ac5be3a74fbde60f5c99457706dd9e802b7f749f0c375a1ecca48424e"
TRANSPORT = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
TRANSPORT_SHA = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
FENCES = {"f12r2n12", "f12r3n03", "f13r1n14", "f13r4n06", "f13r4n09",
          "f12r2n11", "f12r2n13", "f13r1n19", "f13r2n07", "f13r4n03"}
WAVE = "training80-2347-ordinary"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, pin):
    require(not path.is_symlink(), "unexpected evidence link")
    raw = path.read_bytes()
    require(digest(raw) == pin, f"original file changed: {path.name}")
    return raw


def groups():
    plan = json.loads(pinned(CALENDAR, CALENDAR_SHA))
    declared_fences = set(plan["special_scopes_excluded"]) | set(plan["existing_held_fences"])
    rows = [dict(row, previous_worker_pid=row["worker_PID"])
            for group in plan["ordinary_candidates"] for row in group["targets"]]
    visited = {row["node"] for row in rows}
    require(len(visited) == len(rows) == 70 and len({row["previous"] for row in rows}) == 70,
            "ordinary calendar repeats a target")
    require(declared_fences == FENCES and not (FENCES & visited), "protected node in ordinary scope")
    batches = []
    for parent, count in (("1103606", 32), ("1194900", 38)):
        owned = sorted((row for row in rows if row["parent"] == parent),
                       key=lambda row: (row["planned_end_epoch"], row["node"]))
        require(len(owned) == count, "ordinary calendar parent count changed")
        batch = []
        for row in owned:
            end = row["planned_end_epoch"]
            require(re.fullmatch(parent + r"\.[0-9]+", row["previous"])
                    and end == row["worker_start_epoch"] + row["worker_seconds"]
                    and row["prepare_epoch"] == end - 900, "bad actual predecessor binding")
            if batch and (len(batch) == 8 or end - batch[0]["planned_end_epoch"] > 150):
                batches.append(batch)
                batch = []
            batch.append(row)
        if batch:
            batches.append(batch)
    return sorted(batches, key=lambda batch: (batch[0]["planned_end_epoch"], batch[0]["parent"]))


def modules():
    pinned(OLD_WRAPPER, OLD_WRAPPER_SHA)
    pinned(PUBLISHER, PUBLISHER_SHA)
    return runpy.run_path(str(OLD_WRAPPER)), runpy.run_path(str(PUBLISHER))


def remote_reader():
    tree = ast.parse(pinned(OLD_READER, OLD_READER_SHA))
    value, = (ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(name, ast.Name) and name.id == "REMOTE" for name in node.targets))
    compile(value, "<published ordinary terminal reader reuse>", "exec")
    return value


def observe(group, rows, base):
    pinned(TRANSPORT, TRANSPORT_SHA)
    reader = remote_reader()
    helper = runpy.run_path(str(TRANSPORT))["ssh"](REPO.parent)
    argv = [*helper, shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", reader])]
    started = monotonic()
    try:
        result = subprocess.run(argv, input=json.dumps(rows).encode(), capture_output=True, timeout=28)
        rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as err:
        rc, stdout, stderr = None, err.stdout or b"", err.stderr or b""
    require(len(stdout) <= 32768, "terminal stdout exceeds bound")
    observed = json.loads(stdout) if rc == 0 and stdout else None
    data = dict(batch=group, rc=rc, stdout_SHA256=digest(stdout), stderr_SHA256=digest(stderr),
                observed=observed, local_roundtrip_s=monotonic() - started)
    receipt = LOCAL / f"{WAVE}-terminal-group-{group}-{uuid4().hex}.json"
    base["store"](receipt, data)
    released = base["released_nodes"](rows, data)
    require(monotonic() - started < 35, "ordinary terminal receipt aged")
    return receipt, data, released


def child(group, invocation, rows, base, publisher):
    require(os.environ.get("HYGON_STANDBY_VIA_COMPUTE") == "1", "ordinary bridge route changed")
    nodes = os.environ["HYGON_STANDBY_ONLY_NODES"].split()
    require(set(nodes) <= {row["node"] for row in rows}, "child crossed its frozen parent group")
    publisher["child_reservation"](group, nodes, invocation)
    state = json.loads(sys.stdin.buffer.read(8193))
    base["store"](JOURNAL / f"{invocation}-publication-refill-once.json",
                  dict(wrapper_invocation=invocation, batch=group, nodes=nodes,
                       calendar_SHA256=CALENDAR_SHA, replay_allowed=False))
    try:
        publisher["verify_bound"](state, REPO, state["head"], publisher["ALL"])
        action = publisher["render"]()
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        print(json.dumps({"standby_no_dispatch": "publication check before device admission: " + type(err).__name__}))
        return
    os.chdir(REPO.parent)
    check = lambda repo, tip, paths: publisher["verify_bound"](state, repo, tip, paths)
    exec(action, dict(__name__="__main__", certified_tip=state["head"],
                      check_preverified_publication=check, pinned_members=publisher["ALL"]))


def reserve(node, step, intent, base):
    for slot in range(3):
        path = JOURNAL / f"{node}-{step}-{slot}.json"
        try:
            base["store"](path, intent)
        except FileExistsError:
            marker = json.loads(path.read_bytes())
            require(marker["node"] == node and marker["predecessor"] == step
                    and marker["replay_allowed"] is False, "prior reservation changed")
            previous = marker["wrapper_invocation"]
            require(re.fullmatch(r"[a-f0-9]{32}", previous), "bad prior invocation identity")
            outcome = JOURNAL / f"{previous}-outcome.json"
            if not outcome.is_file():
                return False
            row = json.loads(outcome.read_bytes())
            if (row["wrapper_invocation"] != previous or row["no_GPU_send_confirmed"] is not True
                    or node not in row["terminal_ready_nodes"] or row["terminal_receipt"] != marker["terminal_receipt"]):
                return False
            stdout = (JOURNAL / f"{previous}-stdout.private").read_bytes()
            if digest(stdout) != row["stdout_SHA256"] or not base["did_not_send"](stdout, row["refill_rc"]):
                return False
        else:
            return True
    return False


def refill(group, rows, base, publisher):
    require(time() >= rows[0]["planned_end_epoch"] - 65, "first budget is outside single-pass preparation")
    try:
        state = publisher["precheck"]()
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        return dict(batch=group, refill_attempted=False, no_GPU_send_confirmed=True,
                    prior_publication_error=type(err).__name__)
    receipt, observed, released = observe(group, rows, base)
    if not released:
        return dict(batch=group, terminal_receipt=receipt.name, ready_for_device_probe=[],
                    refill_attempted=False, observation_rc=observed["rc"])
    invocation = uuid4().hex
    selected = []
    pin = digest(receipt.read_bytes())
    JOURNAL.mkdir(exist_ok=True)
    for row in released:
        node, step = row["node"], row["previous"]
        intent = dict(batch=group, node=node, predecessor=step, wrapper_invocation=invocation,
                      terminal_receipt=receipt.name, terminal_receipt_SHA256=pin,
                      calendar_SHA256=CALENDAR_SHA, replay_allowed=False, recorded_epoch=time())
        if reserve(node, step, intent, base):
            selected.append(node)
    result = dict(batch=group, terminal_receipt=receipt.name, terminal_receipt_SHA256=pin,
                  terminal_ready_nodes=selected, observation_rc=observed["rc"], refill_attempted=bool(selected))
    if not selected:
        return result
    result.update(wrapper_invocation=invocation, calendar_SHA256=CALENDAR_SHA,
                  publication=state["head"], publication_prechecked_before_terminal=True)
    env = dict(os.environ, HYGON_STANDBY_ONLY_NODES=" ".join(selected), HYGON_STANDBY_VIA_COMPUTE="1")
    argv = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
            "--child", invocation, "--group", str(group)]
    try:
        reply = subprocess.run(argv, input=json.dumps(state).encode(), env=env, cwd=REPO,
                               capture_output=True, timeout=210)
        rc, stdout, stderr = reply.returncode, reply.stdout, reply.stderr
    except subprocess.TimeoutExpired as err:
        rc, stdout, stderr = None, err.stdout or b"", err.stderr or b""
    result.update(refill_rc=rc, no_GPU_send_confirmed=base["did_not_send"](stdout, rc),
                  transport_observation="REFILL_PROCESS_RETURNED" if rc is not None else "UNKNOWN_RECONCILE")
    for stream, raw in (("stdout", stdout), ("stderr", stderr)):
        with (JOURNAL / f"{invocation}-{stream}.private").open("xb") as output:
            output.write(raw)
        result[stream + "_SHA256"] = digest(raw)
    base["store"](JOURNAL / f"{invocation}-outcome.json", result)
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("bridge_launch_pending") is True:
            result["bridge_ack"] = {key: row[key] for key in ("request", "control_label", "requested_cards")}
        if isinstance(row, dict) and row.get("remote_outcome") == "UNKNOWN":
            result["unconfirmed_request"] = row.get("request")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dispatch", action="store_true")
    action.add_argument("--observe-only", action="store_true")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    base, publisher = modules()
    plan = groups()
    if args.child:
        require(args.group is not None and 1 <= args.group <= len(plan), "bad child group")
        child(args.group, args.child, plan[args.group - 1], base, publisher)
        return
    if args.group is None and not args.dispatch and not args.observe_only:
        remote_reader()
        publisher["render"]()
        details = [dict(group=index, parent=rows[0]["parent"], first=rows[0]["planned_end_CST"],
                        last=rows[-1]["planned_end_CST"], nodes=len(rows))
                   for index, rows in enumerate(plan, 1)]
        print(json.dumps(dict(checked=70, groups=details, remote_calls=0)))
        return
    require(args.group is not None and 1 <= args.group <= len(plan), "unknown single group")
    rows = plan[args.group - 1]
    if args.observe_only:
        receipt, observed, released = observe(args.group, rows, base)
        targets = (observed["observed"] or {}).get("targets", [])
        result = dict(group=args.group, receipt=receipt.name, SHA256=digest(receipt.read_bytes()),
                      Slurm={row["step"]: (row["accounting"] or [None, None, None])[2] for row in targets},
                      released=[row["node"] for row in released], rc=observed["rc"], GPU_writes=0)
    elif args.dispatch:
        result = refill(args.group, rows, base, publisher)
    else:
        result = dict(group=args.group, checked=[row["node"] for row in rows], remote_calls=0)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as err:
        print(json.dumps(dict(local_error=type(err).__name__, detail_SHA256=digest(str(err).encode()))))
        raise SystemExit(1)
