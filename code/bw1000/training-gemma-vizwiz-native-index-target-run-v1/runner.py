"""Install and run one authorized Gemma VizWiz zero-GRES index request."""

from __future__ import annotations

import argparse
import ast
from base64 import b64decode
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType


PACKAGE = "code/bw1000/training-gemma-vizwiz-native-index-target-run-v1"
DEPLOYMENT = "code/bw1000/training-gemma-vizwiz-native-index-deployment-v1"
CLOSURE_SHA = "e8e3f68a25c1d16060c9a0b0b6d7f6be8d258fee7cc043176406fc17373fd765"
PROJECT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
VIEW_SHA = "cebeecad13db010a06f2ae7ef48260c6b26cc5ad42ea4993f6b8a60849ce1ad4"
VIEW_BYTES = 345722
TRAINING = "code/bw1000/training-gemma-vizwiz-first-two-updates-v1"
TRAIN_STREAMS = (
    "training-gemma-vizwiz-five-native-input-audit-v1",
    "training-gemma-vizwiz-reviewed-train-native-v1",
)
EPOCH_SECONDS = 120
SCHEMA = "bw1000.gemma-vizwiz.native-index.target-run-v1"
COLUMNS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
CAP = 131072
SHA = re.compile("[0-9a-f]{64}")
REQUEST = re.compile("[0-9a-f]{32}")
NODE = re.compile(r"f[0-9]{2}r[0-9]+n[0-9]{2}")
NUMERIC = re.compile("[0-9]+")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def member(path, maximum, digest=None):
    require(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path.absolute(), "EXACT_MEMBER_PATH")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum and (digest is None or pin(raw)["SHA256"] == digest), "EXACT_MEMBER_BYTES")
    return raw


def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def sync(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def scope_name(request):
    require(REQUEST.fullmatch(request), "EXACT_REQUEST")
    return "gemma-vizwiz-native-index-" + request


def path_arg(value):
    path = PurePosixPath(value)
    require(str(path) == value and path.is_relative_to(PROJECT) and ".." not in path.parts
            and re.fullmatch(r"/[A-Za-z0-9._/-]+", value), "EXACT_TARGET_PATH")
    return path


def decode_packet(raw, digest):
    require(SHA.fullmatch(digest or "") and 0 < len(raw) <= CAP and pin(raw)["SHA256"] == digest, "PACKET_DIGEST")
    packet = json.loads(raw)
    require(set(packet) == {"closure", "files"} and isinstance(packet["files"], dict), "SOURCE_ONLY_PACKET")
    closure_raw = b64decode(packet["closure"], validate=True)
    require(pin(closure_raw)["SHA256"] == CLOSURE_SHA, "SOURCE_CLOSURE_DIGEST")
    closure = json.loads(closure_raw)
    require(set(packet["files"]) == set(closure["members"]) and len(packet["files"]) == 8, "EXACT_SOURCE_MEMBER_SET")
    files = {}
    for name, text in packet["files"].items():
        path = PurePosixPath(name)
        require(str(path) == name and not path.is_absolute() and ".." not in path.parts, "CLOSURE_MEMBER_PATH")
        data = b64decode(text, validate=True)
        require(pin(data) == {key: closure["members"][name][key] for key in ("bytes", "SHA256")}, "CLOSURE_MEMBER_BYTES")
        if path.suffix == ".py":
            ast.parse(data, filename=name, feature_version=(3, 10))
        files[name] = data
    return closure_raw, files, closure


def check_publication(raw, request, closure):
    proof = json.loads(raw)
    require(proof.get("schema") == SCHEMA + "-publication"
            and proof.get("request") == request and proof.get("source_closure_SHA256") == CLOSURE_SHA
            and re.fullmatch("[0-9a-f]{40}", proof["commit"])
            and re.fullmatch("[0-9a-f]{40}", proof["tree"])
            and proof["ref"] == "refs/remotes/origin/bw1000-experiments"
            and proof["commit_verified_in_ref"] is True, "ACTUAL_SOURCE_PUBLICATION_REQUIRED")
    expected = {item["source"]: {key: item[key] for key in ("bytes", "SHA256")} for item in closure["members"].values()}
    expected.update({PACKAGE + "/SOURCE_FILES.json": {"bytes": 2872, "SHA256": CLOSURE_SHA},
                     PACKAGE + "/runner.py": pin(Path(__file__).read_bytes())})
    wanted = proof["members"]
    require(set(wanted) == set(expected) and all(wanted[key] == value for key, value in expected.items()),
            "PUBLICATION_MEMBER_IDENTITIES")
    return proof


def publish_directory(parent, name, write_stage):
    require(parent.is_dir() and parent.resolve(strict=True) == parent.absolute(), "CANONICAL_INSTALL_PARENT")
    target, lock = parent / name, parent / ("." + name + ".install-lock")
    require(not target.exists() and not target.is_symlink(), "REQUEST_ALREADY_INSTALLED")
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise ValueError("REQUEST_ALREADY_INSTALLING") from None
    partial = None
    try:
        require(not target.exists() and not target.is_symlink(), "REQUEST_ALREADY_INSTALLED")
        partial = Path(tempfile.mkdtemp(prefix="." + name + ".pending-", dir=parent)).resolve()
        require(partial.parent == parent.resolve() and partial.stat().st_dev == parent.stat().st_dev,
                "SAME_FILESYSTEM_STAGE")
        result = write_stage(partial)
        sync(partial)
        require(not target.exists() and not target.is_symlink(), "REQUEST_ALREADY_INSTALLED")
        os.rename(partial, target)
        partial = None
        sync(parent)
        return result
    finally:
        if partial is not None:
            require(partial.parent == parent.resolve() and partial.name.startswith("." + name + ".pending-"),
                    "SCOPED_TEMP_CLEANUP")
            shutil.rmtree(partial)
        lock.rmdir()


def verify_view(path):
    raw = member(path, VIEW_BYTES, VIEW_SHA)
    require(len(raw) == VIEW_BYTES and len(raw.splitlines()) == 800, "FROZEN_VIEW_DENOMINATOR")
    return pin(raw)


def install_view(parent, request, raw, expected):
    require(expected == VIEW_SHA and pin(raw) == {"bytes": VIEW_BYTES, "SHA256": VIEW_SHA}
            and len(raw.splitlines()) == 800, "PRIVATE_VIEW_IDENTITY")

    def create(partial):
        view = partial / "train800-swift.jsonl"
        write(view, raw)
        report = {"state": "PRIVATE_VIEW_VERIFIED", "request": request, "view": verify_view(view), "rows": 800}
        write(partial / "PRIVATE_VIEW_RECEIPT.json", encode(report))
        return report

    return publish_directory(parent, scope_name(request) + "-view", create)


def load_source(stage, relative, name):
    manifest = json.loads(member(stage / "SOURCE_FILES.json", 8192, CLOSURE_SHA))["members"]
    identity = manifest[relative]
    exact = stage / relative
    raw = member(exact, identity["bytes"], identity["SHA256"])
    module = ModuleType(name)
    module.__file__ = str(exact)
    exec(compile(raw, str(exact), "exec"), module.__dict__)
    return module


def build_private_view(stage, request, original_paths):
    installed(stage, request)
    builder = load_source(stage, TRAINING + "/build_view.py", "_pinned_gemma_vizwiz_builder")
    scope = builder.load_scope(stage / TRAINING / "TRAINING_SCOPE.json")
    require(tuple(row["original_target_stage"] for row in scope["existing_frozen_rows"]) == TRAIN_STREAMS,
            "EXACT_ORIGINAL_TRAINING_STAGES")
    require(tuple(original_paths) == ("five", "reviewed"), "EXACT_TRAINING_STREAM_SET")
    streams = {}
    for row in scope["existing_frozen_rows"]:
        identity = row["source_stream"]
        streams[row["name"]] = member(original_paths[row["name"]], identity["bytes"], identity["SHA256"])
    view, count = builder.build_view(scope, streams)
    require(count["source_rows"] == count["source_images"] == 12 and count["view_rows"] == 800
            and count["view_bytes"] == VIEW_BYTES and count["view_SHA256"] == VIEW_SHA,
            "FROZEN_BUILT_VIEW_CONTRACT")
    return install_view(stage.parent, request, view, VIEW_SHA)


def target_original_paths():
    return {name: Path(str(PROJECT / "results" / stage / "data/train-swift.jsonl"))
            for name, stage in zip(("five", "reviewed"), TRAIN_STREAMS)}


def install_sources(parent, request, raw, digest, publication_raw, publication_sha):
    closure_raw, files, closure = decode_packet(raw, digest)
    require(pin(publication_raw)["SHA256"] == publication_sha, "PUBLICATION_PIN")
    proof = check_publication(publication_raw, request, closure)

    def create(partial):
        for name, data in files.items():
            path = partial / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, data)
            require(member(path, len(data), pin(data)["SHA256"]) == data, "SOURCE_READBACK")
        write(partial / "SOURCE_FILES.json", closure_raw)
        write(partial / "PUBLICATION.json", publication_raw)
        result = {"state": "SHARED_HOST_SOURCE_STAGE_VERIFIED", "request": request, "packet_SHA256": digest,
                  "closure_SHA256": CLOSURE_SHA, "publication_SHA256": publication_sha,
                  "commit": proof["commit"], "source_members": len(files), "native_cpu_calls": 0, "GPU": 0}
        write(partial / "INSTALL_RESULT.json", encode(result))
        nested = (path for path in partial.rglob("*") if path.is_dir())
        for directory in sorted(nested, key=lambda path: len(path.parts), reverse=True):
            sync(directory)
        return result

    return publish_directory(parent, scope_name(request), create)


def installed(stage, request):
    require(stage.name == scope_name(request) and stage.resolve(strict=True) == stage.absolute()
            and stage.is_dir(), "REQUEST_STAGE_IDENTITY")
    closure = json.loads(member(stage / "SOURCE_FILES.json", 8192, CLOSURE_SHA))
    result_raw = member(stage / "INSTALL_RESULT.json", 8192)
    result = json.loads(result_raw)
    publication_raw = member(stage / "PUBLICATION.json", 8192, result["publication_SHA256"])
    proof = check_publication(publication_raw, request, closure)
    require(result["state"] == "SHARED_HOST_SOURCE_STAGE_VERIFIED" and result["request"] == request
            and result["closure_SHA256"] == CLOSURE_SHA and result["commit"] == proof["commit"], "INSTALLED_REQUEST_SCOPE")
    checked = {}
    for name, item in closure["members"].items():
        raw = member(stage / name, item["bytes"], item["SHA256"])
        require(len(raw) == item["bytes"], "INSTALLED_MEMBER_EXTENT")
        checked[name] = pin(raw)
    return result, pin(result_raw), checked


def original_training_identities(stage, original_paths):
    scope = json.loads(member(stage / TRAINING / "TRAINING_SCOPE.json", 8192))
    require(tuple(original_paths) == ("five", "reviewed"), "EXACT_TRAINING_STREAM_SET")
    seen = {}
    for row in scope["existing_frozen_rows"]:
        identity = row["source_stream"]
        raw = member(original_paths[row["name"]], identity["bytes"], identity["SHA256"])
        require(pin(raw) == identity, "EXACT_TRAINING_STREAM_BYTES")
        seen[row["name"]] = identity
    return seen


def independent_readback(stage, request, view, original_paths, observed):
    result, identity, files = installed(stage, request)
    return {"state": "SHARED_HOST_SOURCE_AND_PRIVATE_VIEW_REHASHED", "request": request,
            "publication_SHA256": result["publication_SHA256"], "commit": result["commit"], "install_result": identity,
            "source_members": files, "view": verify_view(view),
            "original_readonly_train_sources": original_training_identities(stage, original_paths), "observed_epoch": observed,
            "compute_node_checked": False, "native_cpu_calls": 0, "GPU": 0}


def native_modules(stage):
    names = ("index_entry", "render", "verify_terminal")
    previous = {name: sys.modules.pop(name, None) for name in names}
    try:
        modules = []
        for name in names:
            module = load_source(stage, f"{DEPLOYMENT}/{name}.py", name)
            sys.modules[name] = module
            modules.append(module)
        return modules
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def parse_scope(scope, request):
    require(scope["schema"] == SCHEMA + "-scope" and scope["request"] == request
            and scope["root_admitted"] is True and scope["replay_allowed"] is False
            and NUMERIC.fullmatch(scope["parent"]) and NODE.fullmatch(scope["node"])
            and scope["resources"] == {"CPU": 4, "memory_MiB": 8192, "GPU": 0}
            and scope["view_SHA256"] == VIEW_SHA and SHA.fullmatch(scope["placement_receipt_SHA256"]),
            "ACTUAL_CPU_SCOPE")
    require(type(scope["epoch"]) in (int, float) and math.isfinite(scope["epoch"])
            and scope["deadline"] == scope["epoch"] + EPOCH_SECONDS, "EXACT_AFFILIATED_EPOCH")
    for key in ("target_stage", "target_view", "target_output"):
        path_arg(scope[key])
    require(PurePosixPath(scope["target_stage"]).name == scope_name(request)
            and PurePosixPath(scope["target_view"]).name == "train800-swift.jsonl"
            and PurePosixPath(scope["target_output"]).name == scope_name(request) + "-result", "SCOPED_TARGET_NAMESPACE")


def launch_gate(stage, view, original_paths, scope, request, readback, now):
    parse_scope(scope, request)
    require(scope["epoch"] <= now <= scope["epoch"] + 30 and scope["deadline"] - now >= 75,
            "STALE_OR_INSUFFICIENT_CPU_EPOCH")
    require(pin(encode(readback))["SHA256"] == scope["readback_SHA256"]
            and readback["request"] == request
            and 0 <= now - readback["observed_epoch"] <= 90, "FRESH_INDEPENDENT_READBACK")
    actual = independent_readback(stage, request, view, original_paths, readback["observed_epoch"])
    require(actual == readback and actual["publication_SHA256"] == scope["publication_SHA256"], "LAUNCH_READBACK_DIFFERS")
    entry, render, _ = native_modules(stage)
    render.LABEL += "-" + request[:12]
    argv = render.compute_command(scope["parent"], scope["node"], scope["epoch"],
                                  str(PurePosixPath(scope["target_stage"]) / DEPLOYMENT),
                                  str(PurePosixPath(scope["target_view"]).parent), scope["target_output"])
    require(argv.count("--gres=none") == 1 and "--job-name=" + render.LABEL in argv
            and "--cpus-per-task=4" in argv and "--mem=8192M" in argv and entry.LIMIT == EPOCH_SECONDS,
            "EXACT_CPU_DISPATCH_GEOMETRY")
    expected_binds = {
        f"{scope['target_stage']}/{DEPLOYMENT}:/candidate-source:ro",
        f"{PurePosixPath(scope['target_view']).parent}:/vizwiz-view:ro",
        f"{scope['target_output']}:/index-out:rw",
    }
    for stage_name in TRAIN_STREAMS:
        absolute = PROJECT / "results" / stage_name
        expected_binds.add(f"{absolute}:{absolute}:ro")
    binds = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--bind"]
    require(all(binds.count(item) == 1 for item in expected_binds)
            and sum(item.endswith(":rw") for item in binds) == 1 and "--rocm" not in argv,
            "EXACT_READ_ONLY_VIEW_AND_TRAINING_MOUNTS")
    return argv, render.LABEL


def fields(raw):
    return dict(item.split("=", 1) for item in raw.split() if "=" in item)


def query(args, deadline):
    limit = min(6.0, deadline - time.time())
    require(limit > 0, "LOCAL_CONTROL_DEADLINE")
    env = dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S")
    result = subprocess.check_output(args, timeout=limit, env=env)
    require(len(result) <= CAP, "QUERY_OUTPUT_BOUND")
    return result.decode("ascii")


def check_live(scope, label):
    parent, node, deadline = scope["parent"], scope["node"], scope["deadline"] - 65
    jobs = query(["scontrol", "show", "job", parent, "--oneliner"], deadline).splitlines()
    require(len(jobs) == 1, "ACTUAL_PARENT_SCOPE")
    job = fields(jobs[0])
    require(job["JobId"] == parent and job["JobState"] == "RUNNING" and job["UserId"].endswith(f"({os.getuid()})"),
            "ACTUAL_PARENT_SCOPE")
    nodes = query(["scontrol", "show", "hostnames", job["NodeList"]], deadline).splitlines()
    require(node in nodes, "ACTUAL_NODE_MEMBERSHIP")
    live = query(["scontrol", "show", "node", node, "--oneliner"], deadline).splitlines()
    require(len(live) == 1, "ACTUAL_NODE_SCOPE")
    info = fields(live[0])
    require(info.get("NodeName") == node and int(info["CPUTot"]) >= 4 and int(info["RealMemory"]) >= 8192
            and not set(info["State"].split("+")) & {"DOWN", "DRAIN", "FAIL", "MAINT", "NO_RESPOND", "DRAINED"},
            "ACTUAL_CPU_NODE_SCOPE")
    stamp = datetime.fromtimestamp(scope["epoch"] - 120, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    history = query(["sacct", "-nP", "-j", parent, "-S", stamp, "--format=" + COLUMNS], deadline).splitlines()
    require(not any(line.split("|")[1:2] == [label] for line in history), "ORIGINAL_CPU_STEP_ALREADY_EXISTS")


def submit_once(stage, view, original_paths, scope, request, readback, out, *, placement, submit, clock=time.time):
    require(out.parent.is_dir() and out.parent.resolve(strict=True) == out.parent.absolute()
            and not out.exists() and not out.is_symlink(), "REQUEST_ALREADY_SUBMITTED")
    argv, label = launch_gate(stage, view, original_paths, scope, request, readback, clock())
    placement(scope, label)
    require(scope["deadline"] - clock() >= 75, "STALE_OR_INSUFFICIENT_CPU_EPOCH")
    out.mkdir(mode=0o700)
    intent = {"state": "SUBMISSION_POSSIBLE", "request": request, "scope_SHA256": pin(encode(scope))["SHA256"],
              "numeric_step": None, "label": label, "epoch": scope["epoch"], "deadline": scope["deadline"],
              "submitted_epoch": clock(), "argv_SHA256": pin(encode(argv))["SHA256"], "replay_allowed": False}
    write(out / "SUBMISSION_INTENT.json", encode(intent))
    sync(out)
    sync(out.parent)
    result = {"state": "SCHEDULER_TERMINAL_NOT_YET_INDEPENDENTLY_VERIFIED", "request": request,
              "intent": pin(encode(intent)), "planned_train_rows": 12, "qualified_index_rows": 0}
    try:
        result["srun_rc"] = submit(argv, out, scope["deadline"])
    except (OSError, TimeoutError, subprocess.SubprocessError) as error:
        result.update(srun_rc=None, error_type=type(error).__name__)
    write(out / "SUBMISSION_OUTCOME.json", encode(result))
    sync(out)
    return result


def submit(argv, out, deadline):
    with (out / "srun.private.log").open("xb") as stream:
        owner = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            return owner.wait(timeout=max(0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if owner.poll() is not None:
                    break
                try:
                    os.killpg(owner.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    owner.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    continue
            raise


def utc(raw):
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def independent_terminal(stage, request, scope, intent, accounting, step, result=None, index=None):
    parse_scope(scope, request)
    installed(stage, request)
    entry, render, verifier = native_modules(stage)
    label = render.LABEL + "-" + request[:12]
    require(intent["request"] == request and intent["state"] == "SUBMISSION_POSSIBLE"
            and intent["scope_SHA256"] == pin(encode(scope))["SHA256"] and intent["label"] == label
            and intent["replay_allowed"] is False and intent["numeric_step"] is None
            and intent["epoch"] == scope["epoch"] and intent["deadline"] == scope["deadline"]
            and scope["epoch"] <= intent["submitted_epoch"] < scope["deadline"] - 74, "ACTUAL_REQUEST_EPOCH_BINDING")
    require(re.fullmatch(re.escape(scope["parent"]) + r"\.[0-9]+", step), "ACTUAL_NUMERIC_STEP")
    rows = [line.split("|") for line in accounting.splitlines() if line.split("|", 1)[0] == step]
    require(len(rows) == 1 and len(rows[0]) == 9, "ACTUAL_INDEPENDENT_NINE_COLUMN_ACCOUNTING")
    row = rows[0]
    start, end = utc(row[7]), utc(row[8])
    require(row[4].isdecimal() and 0 <= int(row[4]) <= EPOCH_SECONDS and
            math.floor(intent["submitted_epoch"]) <= start <= end <= math.ceil(scope["deadline"])
            and abs(end - start - int(row[4])) <= 1, "ACTUAL_CPU_ELAPSED_OR_DEADLINE")
    render.LABEL = label
    seven_columns = "|".join(row[:7])
    render.terminal(seven_columns, scope["parent"], scope["node"], step)
    if result is None:
        require(index is None, "UNBOUND_INDEX_WITHOUT_WORKER_RESULT")
        verified = {"state": "ACTUAL_ZERO_GRES_TERMINAL_WITHOUT_WORKER_RESULT", "planned_train_rows": 12,
                    "qualified_index_rows": 0}
    else:
        verified = verifier.verify(seven_columns, scope["parent"], scope["node"], step, result, index)
    return dict(verified, request=request, actual_CPU_step=step, full_nine_column_accounting=row,
                epoch=scope["epoch"], deadline=scope["deadline"], intent=pin(encode(intent)), GPU=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("install", "build-view", "readback", "run", "terminal"))
    parser.add_argument("--request", required=True)
    for name in ("packet-sha256", "publication-sha256", "scope-sha256", "step",
                 "independent-accounting-sha256", "placement-sha256"):
        parser.add_argument("--" + name)
    for name in ("publication", "scope", "readback", "view", "accounting", "intent", "worker-result", "index", "placement"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    require(sys.platform == "linux" and sys.flags.dont_write_bytecode, "SHARED_HOST_ONLY")
    os.umask(0o077)
    stage = Path(str(PROJECT / "incoming" / scope_name(args.request)))
    if args.mode == "install":
        result = install_sources(stage.parent, args.request, sys.stdin.buffer.read(CAP + 1), args.packet_sha256,
                                 member(args.publication, 8192, args.publication_sha256), args.publication_sha256)
    elif args.mode == "build-view":
        result = build_private_view(stage, args.request, target_original_paths())
    elif args.mode == "readback":
        require(args.view is not None and path_arg(args.view.as_posix()), "EXPLICIT_PRIVATE_VIEW_PATH")
        result = independent_readback(stage, args.request, args.view, target_original_paths(), time.time())
    else:
        require(SHA.fullmatch(args.scope_sha256 or ""), "ADMITTED_SCOPE_PIN")
        scope = json.loads(member(args.scope, 16384, args.scope_sha256))
        require(pin(encode(scope))["SHA256"] == args.scope_sha256, "CANONICAL_SCOPE_REQUIRED")
        parse_scope(scope, args.request)
        require(scope["target_stage"] == stage.as_posix(), "ACTUAL_TARGET_STAGE")
        if args.mode == "run":
            require(args.view is not None and args.view.as_posix() == scope["target_view"]
                    and args.placement_sha256 == scope["placement_receipt_SHA256"]
                    and SHA.fullmatch(args.placement_sha256 or ""), "LIVE_PLACEMENT_PIN")
            placement = json.loads(member(args.placement, 16384, args.placement_sha256))
            readback = json.loads(member(args.readback, 16384, scope["readback_SHA256"]))
            require(placement["request"] == args.request and placement["parent"] == scope["parent"]
                    and placement["node"] == scope["node"] and placement["CPU_only_admitted"] is True
                    and 0 <= time.time() - placement["observed_epoch"] <= 60, "FRESH_CPU_PLACEMENT_REQUIRED")
            result = submit_once(stage, args.view, target_original_paths(), scope, args.request, readback,
                                 Path(scope["target_output"]),
                                 placement=check_live, submit=submit)
        else:
            require(SHA.fullmatch(args.independent_accounting_sha256 or "")
                    and args.intent == Path(scope["target_output"]) / "SUBMISSION_INTENT.json",
                    "INDEPENDENT_ACCOUNTING_AND_INTENT_REQUIRED")
            out = Path(scope["target_output"])
            summaries = [out / name for name in ("RESULT.json", "FAILED.json") if (out / name).exists()]
            require(len(summaries) <= 1 and args.worker_result == next(iter(summaries), None),
                    "EXACT_SCOPED_WORKER_SUMMARY")
            index_path = out / "native-index.json"
            require(args.index is None or args.index == index_path, "EXACT_SCOPED_NATIVE_INDEX")
            accounting = member(args.accounting, CAP, args.independent_accounting_sha256)
            intent = json.loads(member(args.intent, 16384))
            worker = member(args.worker_result, 16384) if args.worker_result else None
            result = independent_terminal(stage, args.request, scope, intent, accounting.decode("ascii"), args.step,
                                          json.loads(worker) if worker else None,
                                          member(args.index, 16384) if args.index else None)
            result.update(independent_accounting=pin(accounting), worker_summary=pin(worker) if worker else None)
    raw = encode(result)
    print(json.dumps({"result": result, "canonical_receipt": pin(raw)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
