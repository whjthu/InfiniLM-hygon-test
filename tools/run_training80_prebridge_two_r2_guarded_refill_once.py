"""Restore just the two R2 nodes after their original guarded scopes proved no GPU send."""

import argparse
from datetime import datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import sqlite3
import subprocess
from types import FunctionType
from uuid import uuid4


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
STEM = "training80-prebridge-two-r2-guarded-bf16-v2"
CLAIM = REPO / "coordination/tasks" / f"{STEM}.json"
ORIGINAL = REPO / "tools/run_training80_prebridge_five_guarded_refill_once.py"
ORIGINAL_PIN = "7b88d30a4af7eff4f25fbcc663de674af328a03de74e364c517752bd599d306c"
ORIGINAL_CLAIM_PIN = "c4c18f644e6ebc2a99ea0e19d2261a9cbbe47efe743e79dd5ae8f79fea4ce974"
NODES = {"f13r2n03": "1103606.3184", "f13r2n10": "1103606.3185"}
ORIGINAL_FAILURE = "65f3a77095fdcab49b34702824d24e5e44299f56107c36f5985aec601edb2c5b"


HOST_ADAPTER = r'''
def preserve_prior_no_gpu():
    audit = RESULTS / PRIOR_SCOPE / "outcome.json"
    require(audit.resolve(strict=True) == audit, "prior durable outcome path changed")
    with audit.open("rb") as stream:
        original = stream.read(4097)
    require(0 < len(original) <= 4096 and sha256(original).hexdigest() == PRIOR_GUARD_SHA,
            "prior durable no-send bytes changed")
    state = json.loads(original)
    require(state["request"] == PRIOR_REQUEST and state["node"] == NODE and state["parent"] == PARENT
            and state["state"] == "GPU_NOT_SENT" and state["GPU_srun_attempted"] is False
            and state["zero_GRES_call_attempted"] is True and state["probe_rc"] == 0 and state["replay_allowed"] is False,
            "old request does not prove zero GPU send")
    require(not os.path.lexists(RESULTS / ("training80-dcu-standby-" + PRIOR_REQUEST)),
            "prior GPU worker namespace exists")
    row = account(PRIOR_STEP)
    require(row and row[:6] == [PRIOR_STEP, PRIOR_PROBE_LABEL, "COMPLETED", "0:0", NODE, ZERO]
            and row[6] in PRIOR_STEP_END, "prior probe terminal or GRES changed")
    return row


def diagnose_stderr(raw):
    known_tokens = {"task", "tasks", "cpu", "cpus", "affinity", "binding", "allocated", "overlap", "memory", "permission",
                    "denied", "read", "write", "timeout", "cgroup", "plugin", "topology", "step", "stderr", "gres",
                    "dcu", "exclusive", "node", "resource", "resources", "overlap", "restrict", "unsupported"}
    lines = []
    for line in raw[:2048].decode("utf-8", errors="replace").splitlines()[:12]:
        match = re.match(r"^\s*(srun|slurmstepd|apptainer|python3?)(?::|\[)[ \t]*", line, re.I)
        source = match[1].lower() if match else "unrecognized"
        text = line[match.end():] if match else line
        severity = re.match(r"^\s*(warning|error|fatal|notice|info|debug)\b", text, re.I)
        tokens = sorted(set(re.findall(r"[a-z]+", text.lower())) & known_tokens)
        lines.append(dict(source=source, severity=severity[1].lower() if severity else "unrecognized", terms=tokens))
    return dict(classification="NONEMPTY_STDERR_BLOCKED" if raw else "EMPTY", exact_bytes=len(raw),
                exact_SHA256=sha256(raw).hexdigest(), redacted_lines=lines,
                classification_truncated=len(raw) > 2048 or raw[:2048].count(b"\n") >= 12)


def probe_diagnostics(outcome, observation):
    outcome["probe_stderr_diagnostic"] = diagnose_stderr(observation.stderr)
    outcome["probe_stdout_bytes"] = len(observation.stdout)
    if 0 < len(observation.stdout) <= 8192:
        try:
            outcome["unqualified_probe_step_for_reconciliation"] = bind_probe(json.loads(observation.stdout))
        except (ValueError, TypeError, KeyError):
            pass
'''


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def replaced(text, anchor, updated):
    require(text.count(anchor) == 1, "original guarded insertion point differs")
    return text.replace(anchor, updated, 1)


def prepare():
    require(digest(ORIGINAL.read_bytes()) == ORIGINAL_PIN, "published five-node guard changed")
    namespace = runpy.run_path(str(ORIGINAL))
    old_claim = json.loads(namespace["pin"](namespace["CLAIM"], ORIGINAL_CLAIM_PIN))
    config = namespace["prepared"](old_claim)
    raw = namespace["pin"](CLAIM)
    claim = json.loads(raw)
    require((claim["scope"], claim["state"], claim["parent"], claim["GPU_per_node"], claim["GPU_max_total"],
             claim["max_invocations_per_node"], claim["replay_allowed"])
            == (STEM, "PREPARED_ONLY_NO_LIVE_DISPATCH", "1103606", 8, 16, 1, False)
            and set(claim["targets"]) == set(NODES), "exact r2 scope changed")
    new = {item["request"] for item in claim["targets"].values()}
    require(len(new) == 2 and all(re.fullmatch(r"[a-f0-9]{32}", request) for request in new)
            and not new & {item["request"] for item in old_claim["targets"].values()}, "new requests must be independently unique")
    assembly = json.loads(namespace["evidence"](claim["original_terminal"]))
    require(assembly["new_scheduling_operations"] == 0 and assembly["r2_exact_controller_steps"] == []
            and set(assembly["r2"]) == set(NODES), "original independent terminal scope changed")
    prior = {}
    for node, step in NODES.items():
        spec = claim["targets"][node]
        old_request = old_claim["targets"][node]["request"]
        require((spec["original_request"], spec["original_probe_step"]) == (old_request, step), "prior attempt identity changed")
        transport = json.loads(namespace["evidence"](spec["transport"]))["transport"]
        observer = json.loads(namespace["evidence"](spec["durable_read"]))
        observed = observer["observed"]
        guard, guard_sha = observed["original_guard_outcome"]
        require(observer["rc"] == 0 and transport["rc"] == 0 and transport["authority_verified"] is True
                and transport["remote_received"] is True and transport["remote_outcome"] == "RESPONSE_OBSERVED"
                and transport["response"] == guard and observed["scheduler_label_observed"] is False
                and transport["request"] == guard["request"] == old_request and guard["node"] == node
                and guard["state"] == "GPU_NOT_SENT" and guard["GPU_srun_attempted"] is False
                and guard["zero_GRES_call_attempted"] is True and guard["probe_rc"] == 0
                and guard["detail_SHA256"] == ORIGINAL_FAILURE and guard["replay_allowed"] is False
                and re.fullmatch(r"[a-f0-9]{64}", guard_sha), "positive prior no-GPU-send proof differs")
        result = assembly["r2"][node]
        row, = result["probe_rows"]
        label = "training80-g4-fresh-" + old_request[:12]
        require(result["gpu_namespace_exists"] is False and result["gpu_rows"] == [] and row[1] == label
                and namespace["terminal"](row, step, node, namespace["HOST_TRES"]), "prior zero-GRES terminal or namespace differs")
        end = datetime.fromisoformat(row[8])
        require(end.tzinfo is None, "unexpected old accounting time encoding")
        prior[node] = dict(PRIOR_REQUEST=old_request, PRIOR_SCOPE=f"{namespace['STEM']}-{node}-{old_request}",
                           PRIOR_GPU_LABEL=f"training80-dcu-standby-{old_request[:12]}-{node}",
                           PRIOR_GUARD_SHA=guard_sha, PRIOR_STEP=step, PRIOR_PROBE_LABEL=label,
                           PRIOR_STEP_END=[end.isoformat(), (end + timedelta(hours=8)).isoformat()])
    return namespace, config, prior, old_claim, claim, digest(raw)


def render(node, prepared, publication):
    original, config, prior, old_claim, claim, _ = prepared
    request = claim["targets"][node]["request"]
    new_claim = dict(old_claim, targets={**old_claim["targets"], node: dict(old_claim["targets"][node], request=request)})
    program = original["render"](node, new_claim, config, publication)
    before = f"SCOPE={original['STEM'] + '-' + node + '-' + request!r}"
    new_scope = f"SCOPE={STEM + '-' + node + '-' + request!r}"
    program["observe"] = replaced(program["observe"], before, new_scope)
    launch = replaced(program["launch"], before, new_scope)
    launch = replaced(launch, "\ndef run_guarded():", HOST_ADAPTER + "\ndef run_guarded():")
    launch = replaced(launch, "([old_label], [gpu_label])", "([old_label], [gpu_label], [PRIOR_GPU_LABEL])")
    anchor = "            check_previous(account(PREVIOUS))"
    launch = replaced(launch, anchor, "            outcome['prior_no_gpu_probe_terminal'] = preserve_prior_no_gpu()\n" + anchor)
    gate = "            require(observation.returncode == 0 and not observation.stderr and 0 < len(observation.stdout) <= 8192,"
    launch = replaced(launch, gate, "            probe_diagnostics(outcome, observation)\n" + gate)
    program["launch"] = "\n".join(f"{key}={value!r}" for key, value in prior[node].items()) + "\n" + launch
    compile(program["launch"], "<new guarded production probe>", "exec")
    compile(program["observe"], "<new guarded readback>", "exec")
    return program


def adapted(original, name):
    # Run the existing guarded local operation with a new receipt namespace only.
    function = original[name]
    bound = dict(function.__globals__, STEM=STEM)
    paths = original["paths"]
    bound["paths"] = FunctionType(paths.__code__, dict(paths.__globals__, STEM=STEM))
    return FunctionType(function.__code__, bound, name)


def publication(revision, original, config):
    original["verify_publication"](revision, config)
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
    for path in (Path(__file__).resolve(), CLAIM):
        blob = subprocess.check_output([*git, "cat-file", "blob", revision + ":" + path.relative_to(REPO).as_posix()],
                                       stderr=subprocess.PIPE, cwd=REPO, timeout=10)
        require(blob == path.read_bytes(), "new published guarded source or claim differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--publication")
    modes.add_argument("--observe-original", action="store_true")
    parser.add_argument("--node", action="append", choices=sorted(NODES))
    args = parser.parse_args()
    nodes = args.node or (sorted(NODES) if args.check else [])
    require(nodes and len(set(nodes)) == len(nodes), "choose each exact node at most once")
    prepared = prepare()
    original, config, _, _, _, claim_pin = prepared
    source = config[0]
    original["pin"](source["HELPER"], source["HELPER_SHA"])
    original["pin"](source["TRANSPORT"], source["TRANSPORT_SHA"])
    programs = {n: render(n, prepared, args.publication) for n in nodes}
    if args.check:
        print(json.dumps(dict(claim_SHA256=claim_pin, GPU=0, Slurm=0, SCNet=0, Git=0,
                              nodes={n: dict(request=p["request"], launch_SHA256=digest(p["launch"].encode()),
                                            observe_SHA256=digest(p["observe"].encode()), worker_seconds=p["worker_seconds"])
                                     for n, p in programs.items()}), sort_keys=True))
        return 0
    if args.publication:
        publication(args.publication, original, config)
        claimed = original["live_authority"](config)
        operation = adapted(original, "dispatch")
    else:
        operation = adapted(original, "observe")
    failures = 0
    for node, program in programs.items():
        try:
            result = (operation(node, program, config) if args.observe_original else
                      operation(node, program, args.publication, claim_pin, config, claimed))
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, subprocess.SubprocessError) as error:
            failures += 1
            path = LOCAL / f"{STEM}-{node}-failure-{uuid4().hex}.json"
            result = dict(node=node, request=program["request"], state="REJECTED_OR_RECONCILE_ORIGINAL",
                          error_class=type(error).__name__, error_SHA256=digest(str(error).encode()), replay_allowed=False)
            config[1]["save"](path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(json.dumps(dict(local_error=type(error).__name__, error_SHA256=digest(str(error).encode()))))
        raise SystemExit(1)
