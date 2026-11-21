"""Admit the frozen Kimi r461 score compile once, or reconcile its original step."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import struct
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "Kimi461-output-residual-wave-score-gfx936-CPU-v1"
PARENT, NODE, ZONE = "1103606", "f12r2n11", "O01"
PARENTS = ("1087656", PARENT)
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
PACKAGE = HERE.parent / "output-residual-wave-compile-CPU-v1"
PACKAGE_PIN = "e55041e1f3b5b48c776ce0a3983fce550559dbdc0daa4a50e4dacd4319b13efe"
MANIFEST = "d2544aeaaca73ff6683dda62fc82b097e02bf28cf9dde1ec0a720fe9e1b52df6"
REQUEST = sha256(f"{LABEL}:{PARENT}:{NODE}:{MANIFEST}".encode()).hexdigest()[:32]
MEMBERSHIP = "tools/read_training_slurm_membership.py"
LEDGER = "coordination/training/training80-active-node-task-map-v1.json"
ACCOUNTING = "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py"
PINS = {
    MEMBERSHIP: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    LEDGER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    ACCOUNTING: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
}
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
ARMS = ("full-tree", "wave32")
SYMBOL = "kimi_output_residual_scores_v1_launch"
PRODUCER = "Deep-C4-constant-slots-r482-Native-CPU-v1"
INSIDE_SOURCE = PurePosixPath("/refactor/incoming") / LABEL
INSIDE_OUTPUT = PurePosixPath("/refactor/candidates") / LABEL


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path, pin=None, limit=2**21):
    require(path.resolve(strict=True) == path, f"noncanonical evidence: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (pin is None or sha256(raw).hexdigest() == pin),
            f"evidence bytes differ: {path.name}")
    return raw


def save_once(path, row):
    raw = (json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(raw).hexdigest()


def frozen(claim_sha):
    claim = json.loads(read(HERE / "CLAIM.json", claim_sha, 4096))
    required = dict(request=REQUEST, label=LABEL, placement=PLACEMENT, wall_seconds=300,
                    gpu_target="gfx936", planned_device_TUs=2, rendered_manifest_SHA256=MANIFEST,
                    frozen_package_manifest_SHA256=PACKAGE_PIN, replay_allowed=False, GPU_GO=False, Model_GO=False)
    require(all(type(claim[key]) is type(value) and claim[key] == value for key, value in required.items()),
            "Kimi score claim differs")
    members = json.loads(read(PACKAGE / "SOURCE_FILES.json", PACKAGE_PIN))["files_SHA256"]
    for name, pin in members.items():
        require(Path(name).name == name, "unexpected frozen package member")
        read(PACKAGE / name, pin)
    raw = {name: read(REPO / name, pin) for name, pin in PINS.items()}
    member, terminal = {}, {}
    for name, namespace in ((MEMBERSHIP, member), (ACCOUNTING, terminal)):
        namespace.update(__name__="frozen_slurm_primitives")
        exec(compile(raw[name], str(REPO / name), "exec"), namespace)
    ledger = json.loads(raw[LEDGER])
    groups = {parent: {name: group["allocations"][parent] for name, group in ledger["groups"].items()}
              for parent in PARENTS}
    require(NODE in groups[PARENT][ZONE], "Kimi score node is outside approved O01")
    return member, terminal, groups, runpy.run_path(str(PACKAGE / "prepare.py"))["prepare"]


def generated_payload(path):
    raw = read(path / "SOURCE_FILES.json", MANIFEST, 8192)
    files = json.loads(raw)["files_SHA256"]
    require(len(files) == 13, "rendered score member count differs")
    for name, pin in files.items():
        read(path / name, pin)
    return files


def parse_accounting(raw):
    require(len(raw) <= 2**21, "oversized accounting result")
    rows = [line.split("|") for line in raw.decode().splitlines() if line]
    require(all(len(row) == len(COLUMNS) for row in rows), "unexpected accounting columns")
    matches = [line for line in rows if line[1] == LABEL]
    require(len(matches) <= 1, "original score label resolves to several steps")
    row = matches[0] if matches else None
    require(row is None or (re.fullmatch(PARENT + r"\.[0-9]+", row[0]) and row[2].strip() and row[5] == NODE),
            "original score step reported a different parent or node")
    return row


def accounting(timeout):
    return parse_accounting(subprocess.check_output(
        ["sacct", "-n", "-P", "-j", ",".join(PARENTS), "-S", "2026-11-20T00:00:00", "-o", ",".join(COLUMNS)],
        timeout=timeout))


def placement(snapshot, now):
    parents = snapshot["parents"]
    epoch = snapshot["observed_epoch"]
    require(set(parents) == set(PARENTS) and snapshot["membership_qualified"] is True
            and all(row["owned_by_observer"] is True for row in parents.values())
            and 0 <= now - epoch <= 10, "invalid fresh owned job membership")
    node = parents[PARENT]["coverage_by_group_and_node"][ZONE][NODE]
    require(node["state"] == "NO_NUMBERED_STEP" and node["numbered_step_assigned_DCU_bounds"] == [0, 0],
            "O01 node already has a numbered step")
    return dict(observed_epoch=epoch, parent=PARENT, node=NODE, state=node["state"])


def parent_expiry(member):
    lines = member["query"]("scontrol", "show", "job", PARENT, "--oneliner")
    require(len(lines) == 1, "original allocation is unreadable")
    row = member["fields"](lines[0])
    require(row.get("JobId") == PARENT and row.get("JobState") == "RUNNING"
            and re.fullmatch(r"[^()]+\(" + str(os.getuid()) + r"\)", row.get("UserId", "")),
            "original allocation changed owner")
    return datetime.fromisoformat(row["EndTime"]).timestamp()


def command(root, generated, epoch):
    env = {"TUNING_CPU_PARENT": PARENT, "TUNING_CPU_NODE": NODE, "TUNING_CPU_MANIFEST_SHA": MANIFEST,
           "TUNING_CPU_EPOCH": repr(epoch), "TUNING_CPU_DEADLINE": repr(epoch + 300),
           "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1",
           "ROCM_PATH": "/opt/dtk", "LC_ALL": "C"}
    worker = ["env", *(f"{key}={value}" for key, value in env.items()), "/usr/bin/python3", "-I", "-S", "-B",
              "-u", str(INSIDE_SOURCE / "compile_once.py")]
    shell = ("set -euo pipefail\nulimit -c 0\nsource /opt/dtk/env.sh\n"
             "unset LD_PRELOAD LD_AUDIT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH\n"
             f"budget=$(({math.floor(epoch + 300)} - $(/usr/bin/date +%s) - 20))\n"
             '((budget > 1)) || exit 124\nexec /usr/bin/timeout --signal=TERM --kill-after=5s "${budget}s" '
             + shlex.join(worker))
    binds = [(root, "/refactor", "ro"), (root / "candidates", "/refactor/candidates", "rw"),
             (root / "candidates" / PRODUCER, f"/refactor/candidates/{PRODUCER}", "ro"),
             (generated, str(INSIDE_SOURCE), "ro"), (Path("/opt/hyhal"), "/opt/hyhal", "ro")]
    apptainer = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--no-home", "--pwd", "/"]
    for source, target, mode in binds:
        apptainer.extend(("--bind", f"{source}:{target}:{mode}"))
    return ["srun", f"--jobid={PARENT}", "--exact", "--immediate=5", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=4", "--cpu-bind=threads", "--mem=8192M", "--gres=none", f"--nodelist={NODE}",
            "--time=00:05:00", f"--job-name={LABEL}", *apptainer, str(root / "containers/dtk26-sandbox-r1"),
            "/bin/bash", "--noprofile", "--norc", "-c", shell]


def dep_paths(raw, obj):
    left, colon, right = raw.decode().replace("\\\n", " ").partition(":")
    names = set(shlex.split(right))
    require(colon and shlex.split(left) == [str(obj)] and names and all(name.startswith("/") for name in names),
            "original M/MD target differs")
    return names


def compile_commands(plan):
    flags = plan["compiler"]["compiler_prefix"]
    preprocess, builds = [], []
    for arm in ARMS:
        src = str(INSIDE_SOURCE / arm / "fused_scores.cu")
        work = INSIDE_OUTPUT / arm
        obj = str(work / "fused_scores.o")
        preprocess.append((arm + "-M", flags + ["-M", "-MF", str(work / "fused_scores.before.d"), "-MT", obj, src]))
        builds.extend(((arm + "-compile", flags + ["-MD", "-MF", str(work / "fused_scores.d"), "-MT", obj, "-c",
                                                  src, "-o", obj]),
                       (arm + "-symbols", ["/usr/bin/nm", "-g", "--defined-only", obj])))
    return preprocess + builds


def terminal_result(row, epoch, generated, output, member):
    if row[2:4] != ["COMPLETED", "0:0"]:
        return None
    info = member["completed_numbered_step"](COLUMNS, row, row[0], scope="Kimi score compile")
    res = dict(field.split("=", 1) for field in info["AllocTRES"].split(","))
    require(res.get("cpu") == "4" and res.get("node") == "1" and res.get("mem") in ("8G", "8192M")
            and member["allocated_dcu"](info["AllocTRES"]) == 0 and row[4].isdecimal() and int(row[4]) <= 300,
            "original score step resources differ")
    pins = generated_payload(generated)
    plan = json.loads(read(generated / "BUILD_PLAN.json", pins["BUILD_PLAN.json"]))
    raw = read(output / "CPU-result.json")
    receipt = json.loads(raw)
    expected = dict(original_step=row[0], node=NODE, manifest_SHA256=MANIFEST, epoch=epoch, deadline=epoch + 300,
                    planned_device_TUs=2, actual_device_TUs=2, actual_GPU_steps=0, runtime_calls=0,
                    all_fresh_M_before_compile=True, passed=True, completed=True, device_qualified=False)
    require(all(type(receipt[key]) is type(value) and receipt[key] == value for key, value in expected.items())
            and set(receipt["arms"]) == set(ARMS), "original score worker scope differs")
    commands = receipt["commands"]
    require([(cmd["label"], cmd["argv"]) for cmd in commands] == compile_commands(plan)
            and all(type(cmd["rc"]) is int and cmd["rc"] == 0 for cmd in commands),
            "original preprocessor/compile sequence differs")
    command_by_label = {cmd["label"]: cmd for cmd in commands}
    shares, verified = [], {}
    for arm in ARMS:
        record = receipt["arms"][arm]
        obj = INSIDE_OUTPUT / arm / "fused_scores.o"
        paths = output / arm
        object_bytes = read(paths / obj.name, record["object_SHA256"], 2**24)
        before = read(paths / "fused_scores.before.d", record["fresh_M_SHA256"])
        after = read(paths / "fused_scores.d", record["actual_MD_SHA256"])
        owned = {str(INSIDE_SOURCE / f"{arm}/{name}"): pin for name, pin in plan["rendered_files_SHA256"][arm].items()
                 if name in ("fused_scores.cu", "score_math.hpp", "reduction.hpp")}
        deps = record["dependencies_SHA256"]
        require(dep_paths(before, obj) == dep_paths(after, obj) == set(deps)
                and all(deps.get(name) == pin for name, pin in owned.items())
                and deps.get(plan["compiler"]["cuda_runtime_header"]) == plan["compiler"]["cuda_runtime_header_SHA256"]
                and all(re.fullmatch("[a-f0-9]{64}", pin) for pin in deps.values()), "independent M/MD drift")
        nm = command_by_label[arm + "-symbols"]
        log = read(output / (nm["label"] + ".stdout"), nm["stdout_SHA256"])
        exports = {row[-1] for line in log.decode().splitlines() if len(row := line.split()) >= 3
                   and row[-2] == "T" and row[-1].startswith("kimi_output_residual_")}
        require(record["object"] == str(obj) and record["exports"] == [SYMBOL] and exports == {SYMBOL}
                and len(object_bytes) >= 64 and object_bytes[:6] == b"\x7fELF\x02\x01"
                and struct.unpack_from("<HH", object_bytes, 16) == (1, 62), "original score object or export differs")
        shares.append({name: pin for name, pin in deps.items() if name not in owned})
        verified[arm] = {key: record[key] for key in ("object_SHA256", "fresh_M_SHA256", "actual_MD_SHA256")}
        verified[arm]["symbols_stdout_SHA256"] = nm["stdout_SHA256"]
    require(shares[0] == shares[1] and verified[ARMS[0]]["object_SHA256"] != verified[ARMS[1]]["object_SHA256"],
            "score arms do not have a common DTK closure and two distinct objects")
    return dict(score_receipt_SHA256=sha256(raw).hexdigest(), arms=verified)


def outcome(row, epoch, generated, output, member, is_terminal):
    if not is_terminal(row):
        return dict(state="UNKNOWN")
    result = dict(state="TERMINAL_UNVERIFIED", columns=COLUMNS, row=row,
                  terminal_row_SHA256=sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest())
    try:
        verified = terminal_result(row, epoch, generated, output, member)
        if verified is not None:
            result.update(verified, state="COMPILE_VERIFIED")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    return result


def launch(root, scope, output, claim_sha, *, member, terminal, render, observe, expiry, query=accounting,
           run=subprocess.run, wall=time.time, monotonic=time.monotonic):
    require(not any(path.exists() or path.is_symlink() for path in (scope, output)), "original namespace exists")
    ends = expiry()
    require(query(8) is None, "original score step already exists")
    slot = placement(observe(), wall())
    require(ends - wall() >= 300, "original allocation expires within the compile window")
    scope.mkdir(mode=0o700)
    epoch, started = wall(), monotonic()
    generated = scope / "generated"
    intent = dict(request=REQUEST, label=LABEL, claim_SHA256=claim_sha, source_manifest_SHA256=MANIFEST,
                  placement=PLACEMENT, epoch=epoch, deadline=epoch + 300, preflight=slot, parent_end_epoch=ends,
                  replay_allowed=False)
    pin = save_once(scope / "intent.json", intent)
    result = dict(request=REQUEST, state="PRELAUNCH_REFUSED", intent_SHA256=pin, exec_request_attempted=False,
                  observations=[], actual_GPU_steps=0)

    def left():
        return min(epoch + 300 - wall(), 300 - (monotonic() - started))

    def observe_accounting():
        require(left() > 2, "original accounting budget expired")
        row = query(min(left() - 1, 8))
        result["observations"].append(row)
        return row

    try:
        require(render(generated) == dict(manifest_SHA256=MANIFEST, files=13, target_compiles=0),
                "materialized manifest differs from the downstream link admission")
        generated_payload(generated)
        require(left() > 40 and 0 <= wall() - slot["observed_epoch"] <= 10 and ends >= epoch + 300,
                "original launch budget or ownership observation expired")
        argv = command(root, generated, epoch)
        with (scope / "srun.log").open("xb") as log:
            save_once(scope / "exec-attempt.json", dict(request=REQUEST, intent_SHA256=pin, argv=argv,
                                                         attempted_epoch=wall(), replay_allowed=False))
            result.update(state="UNKNOWN", exec_request_attempted=True)
            try:
                result["client_rc"] = run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=left() - 25, check=False).returncode
            except subprocess.TimeoutExpired:
                result["client_timeout"] = True
        seen = terminal["wait_terminal"](observe_accounting, monotonic() + min(20, left() - 2), max_observations=6)
        result.update(outcome(seen["row"], epoch, generated, output, member, terminal["is_terminal"]))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    finally:
        save_once(scope / "accounting-result.json", result)
    return result


def reconcile(scope, output, claim_sha, member, terminal, query=accounting):
    raw = read(scope / "intent.json", limit=16384)
    intent, pin = json.loads(raw), sha256(raw).hexdigest()
    epoch = intent["epoch"]
    require(intent["request"] == REQUEST and intent["label"] == LABEL and intent["placement"] == PLACEMENT
            and intent["source_manifest_SHA256"] == MANIFEST and intent["claim_SHA256"] == claim_sha
            and type(epoch) is float and math.isfinite(epoch) and epoch > 0 and intent["deadline"] == epoch + 300
            and intent["replay_allowed"] is False, "unbound original score intent")
    attempt = json.loads(read(scope / "exec-attempt.json", limit=16384))
    require(attempt["request"] == REQUEST and attempt["intent_SHA256"] == pin and attempt["replay_allowed"] is False
            and attempt["argv"] == command(scope.parents[1], scope / "generated", epoch)
            and epoch <= attempt["attempted_epoch"] < epoch + 300, "unbound original score attempt")
    row = query(8)
    prior = scope / "accounting-result.json"
    if prior.exists():
        old = json.loads(read(prior))
        require(old["request"] == REQUEST and old["intent_SHA256"] == pin, "unbound saved score result")
        former = old.get("row")
        require(former is None or (old["columns"] == COLUMNS and former == row), "original terminal accounting changed")
    return dict(request=REQUEST, intent_SHA256=pin, independent_observation=True, observations=[row],
                **outcome(row, epoch, scope / "generated", output, member, terminal["is_terminal"]))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--reconcile", action="store_true")
    options = parser.parse_args(args)
    require(REPO.is_relative_to(ROOT) and ROOT.resolve(strict=True) == ROOT,
            "expected canonical checkout in the original project")
    member, terminal, groups, render = frozen(options.claim_sha256)
    scope = ROOT / "results" / (LABEL + "-controller")
    out = ROOT / "candidates" / LABEL
    if options.reconcile:
        result = reconcile(scope, out, options.claim_sha256, member, terminal)
    else:
        result = launch(ROOT, scope, out, options.claim_sha256, member=member, terminal=terminal, render=render,
                        observe=lambda: member["snapshot"](groups, owner_uid=os.getuid()),
                        expiry=lambda: parent_expiry(member))
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(result["state"] != "COMPILE_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
