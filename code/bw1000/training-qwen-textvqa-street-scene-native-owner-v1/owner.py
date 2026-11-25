"""Install the pinned private street-scene slice and own one admitted CPU step."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import time
from zipfile import ZipFile


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
APPTAINER = Path("/public/software/apps/apptainer/1.3.4/bin/apptainer")
RECEIPT_SHA = "b5e2563b1437bec31b24f32ee546b1e8fa73b2bd64f9e52e7b9cf60a70558743"
ZIP_SHA = "ba880f5e08551dcdefb7a83e866f682781c8638aa7efada8fd5a0c744e2a8df6"
DATA_SHA = "dd6c882665db32a47ff4736392b63f0d83e4ea0caa226534ec2f1093c6c04ecc"
MEDIA_SHA = "0b2ff0a69ccee6bde9d91445b66754de46b80691a1bf93e5d1fea74d3450947d"
CONFIG_SHA = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
TEMPLATE_SHA = "d4a35ced472f7b10eb254cc334ec21db44f0793a750b68d5edeab105c7db15c3"
VERSIONS = {"ms-swift": "4.4.2", "transformers": "5.12.1", "qwen-vl-utils": "0.0.14"}
ZIP_BYTES, RECEIPT_BYTES, MEMBERS, ROWS = 5355621, 2980, 22, 41
CPUS, MEMORY, LIMIT = 8, 12288, 480
PILOT = "code/bw1000/training-domain-pilot-v1"
SOURCES = {
    "src/build_textvqa_street_scene.py": "172f51c9f2014505a79fbb4c2c556a93a3c771c6076d65d56c842f13b1c526b6",
    "src/check_qwen_native_inputs.py": "044b3ba93f119ead714b77573265ea4c8d1705258ac02af6b177a045aa4f8bba",
    "src/check_textvqa_street_scene_native.py": "0ee5897963121c600b98339e48269a194cb9d45f0087c94d8970acf70597c02f",
}
LABEL = "training-qwen-textvqa-native-input-CPU-"
ACCOUNTING = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
WORKER = """import os, re, runpy, socket, sys
parent, node, checker, *args = sys.argv[1:]
e = os.environ
step = e.get("SLURM_STEP_ID", "")
if (e.get("SLURM_JOB_ID") != parent or not re.fullmatch("[0-9]+", step)
    or socket.gethostname().split(".")[0] != node
    or e.get("SLURM_STEP_GPUS") not in (None, "", "(null)")
    or e.get("SLURM_GPUS_ON_NODE") not in (None, "", "(null)", "0")
    or any(e.get(k) != "" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))):
    raise SystemExit("NUMERIC_ZERO_GRES_STEP_ENV_REQUIRED")
print("EXACT_STEP=" + parent + "." + step, flush=True)
sys.path.insert(0, os.path.dirname(checker))
sys.argv = [checker, *args]
runpy.run_path(checker, run_name="__main__")
"""


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def pattern(value, regex):
    return isinstance(value, str) and re.fullmatch(regex, value) is not None


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def regular(path):
    return path.is_file() and path.resolve(strict=True) == path.absolute() and not path.is_symlink()


def directory(path):
    return path.is_dir() and path.resolve(strict=True) == path.absolute() and not path.is_symlink()


def read(path, limit, *, expected=None, length=None):
    need(regular(path), "EXACT_REGULAR_INPUT")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and (length is None or len(raw) == length)
         and (expected is None or digest(raw) == expected), "EXACT_INPUT_BYTES")
    return raw


def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def namespace(results, request):
    need(directory(results) and pattern(request, "[0-9a-f]{32}"), "EXACT_OWNER_NAMESPACE")
    return results / (LABEL + request)


def manifest(receipt_path):
    receipt = json.loads(read(receipt_path, RECEIPT_BYTES, expected=RECEIPT_SHA, length=RECEIPT_BYTES))
    members = receipt["members"]
    names = set(members)
    images = {name for name in names if pattern(name, "[0-9a-f]{16}\\.jpg")}
    need(len(images) == 16 and len(members) == MEMBERS
         and names == images | SOURCES.keys() | {"TRAIN_SWIFT.jsonl", "MEDIA_PINS.json", "ATTRIBUTION.json"}
         and (receipt["data_rows"], receipt["source_members"], receipt["expected_GPU"],
              receipt["zip_SHA256"], receipt["zip_bytes"]) == (ROWS, 3, 0, ZIP_SHA, ZIP_BYTES)
         and members["TRAIN_SWIFT.jsonl"]["SHA256"] == DATA_SHA
         and members["MEDIA_PINS.json"]["SHA256"] == MEDIA_SHA
         and all(members[name]["SHA256"] == sha for name, sha in SOURCES.items()),
         "EXACT_ORIGINAL_22_MEMBER_MANIFEST")
    return members


def unpack(raw, members):
    with ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        need(len(entries) == len(members) and set(archive.namelist()) == set(members), "EXACT_ARCHIVE_MEMBER_SET")
        output = {}
        for item in entries:
            name = item.filename
            member = members[name]
            path = PurePosixPath(name)
            mode = stat.S_IFMT(item.external_attr >> 16)
            need(not path.is_absolute() and ".." not in path.parts and "\\" not in name
                 and item.orig_filename == name and not item.is_dir()
                 and mode in (0, stat.S_IFREG) and not item.flag_bits & 1
                 and item.file_size == member["bytes"], "EXACT_SAFE_ARCHIVE_ENTRY")
            with archive.open(item) as stream:
                content = stream.read(member["bytes"] + 1)
            need(len(content) == member["bytes"] and digest(content) == member["SHA256"], "EXACT_ARCHIVE_MEMBER_BYTES")
            output[name] = content
        return output


def inputs(root, members):
    need(directory(root) and directory(root / "src"), "EXACT_STAGE_DIRECTORY")
    seen = {path.relative_to(root).as_posix() for path in root.rglob("*") if path != root / "src"}
    need(seen == set(members), "EXACT_INSTALLED_MEMBER_SET")
    for name, pin in members.items():
        read(root / name, pin["bytes"], expected=pin["SHA256"], length=pin["bytes"])


def stage(results, request, archive, receipt):
    run = namespace(results, request)
    members = manifest(receipt)
    content = unpack(read(archive, ZIP_BYTES, expected=ZIP_SHA, length=ZIP_BYTES), members)
    run.mkdir(mode=0o700)
    pending = run / "installing"
    pending.mkdir(mode=0o700)
    (pending / "src").mkdir(mode=0o700)
    for name, raw in content.items():
        save(pending / name, raw)
    inputs(pending, members)
    pending.rename(run / "inputs")
    inputs(run / "inputs", members)
    save(run / "private-transfer-receipt.json", read(receipt, RECEIPT_BYTES,
                                                    expected=RECEIPT_SHA, length=RECEIPT_BYTES))
    report = dict(request=request, state="EXACT_PRIVATE_INPUTS_INSTALLED", members=MEMBERS, planned=ROWS,
                  zip_SHA256=ZIP_SHA, transfer_receipt_SHA256=RECEIPT_SHA, actual_native_cpu_rows=0,
                  scheduler_operations=0, GPU=0, replay_allowed=False)
    save(run / "staged.json", encode(report))
    return report


def installed(run, request):
    need(directory(run), "EXACT_INSTALLED_OWNER")
    record = json.loads(read(run / "staged.json", 2048))
    need(record["request"] == request and record["zip_SHA256"] == ZIP_SHA
         and record["transfer_receipt_SHA256"] == RECEIPT_SHA and record["members"] == MEMBERS
         and record["state"] == "EXACT_PRIVATE_INPUTS_INSTALLED", "EXACT_STAGE_RECEIPT")
    members = manifest(run / "private-transfer-receipt.json")
    inputs(run / "inputs", members)
    return members


def placement(claim, request, publication):
    need(claim["request"] == request and claim["publication"] == publication
         and pattern(publication, "[0-9a-f]{40}") and pattern(claim["parent"], "[0-9]+")
         and pattern(claim["node"], "f[0-9]+r[0-9]+n[0-9]+")
         and claim["CPU_GO"] is True and claim["GPU_GO"] is False and claim["exclusive_node"] is True
         and claim["replay_allowed"] is False and claim["zip_SHA256"] == ZIP_SHA
         and claim["owner_SHA256"] == digest(Path(__file__).read_bytes())
         and claim["dataset_SHA256"] == DATA_SHA
         and (claim["cpus"], claim["memory_MiB"], claim["allocated_DCU"],
              claim["max_deadline_s"], claim["planned"]) == (CPUS, MEMORY, 0, LIMIT, ROWS),
         "EXACT_CPU_OWNER_PLACEMENT")


def fresh(authority, claim, now):
    observed = authority["observed_epoch"]
    need(authority["request"] == claim["request"] and authority["parent"] == claim["parent"]
         and authority["node"] == claim["node"] and authority["publication"] == claim["publication"]
         and authority["exclusive_cpu_admit"] is True and authority["GPU"] == 0
         and pattern(authority["coordinator_proof_SHA256"], "[0-9a-f]{64}")
         and type(observed) in (int, float) and 0 <= now - observed < 60,
         "FRESH_EXCLUSIVE_COORDINATOR_PROOF")


def query(*argv):
    raw = subprocess.check_output(argv, stdin=subprocess.DEVNULL, timeout=8,
                                  env=dict(os.environ, TZ="UTC", SLURM_TIME_FORMAT="%Y-%m-%dT%H:%M:%S"))
    need(len(raw) <= 1048576, "SCHEDULER_RESPONSE_EXTENT")
    return raw.decode("ascii").splitlines()


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def one(rows):
    need(len(rows) == 1, "EXACT_SINGLE_SCHEDULER_ROW")
    return rows[0]


def hosts(name, query_job):
    need(pattern(name, r"[A-Za-z0-9,\[\]._-]+") and name not in ("None", "Unknown"), "EXACT_NODE_LIST")
    result = query_job("scontrol", "show", "hostnames", name)
    need(result and len(result) == len(set(result)), "EXACT_SCHEDULER_NODE_EXPANSION")
    return result


def parent_resources(parent, node, query_job):
    records = [(ids, mem) for line in query_job("scontrol", "-d", "show", "job", parent)
               for nodes, ids, mem in re.findall(r"\bNodes=(\S+)\s+CPU_IDs=(\S+)\s+Mem=(\S+)", line)
               if node in hosts(nodes, query_job)]
    cpu_map, mem = one(records)
    slots = set()
    for item in cpu_map.split(","):
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", item)
        need(match is not None, "EXACT_PARENT_CPU_MAP")
        first, last = int(match[1]), int(match[2] or match[1])
        need(0 <= first <= last <= 16384, "EXACT_PARENT_CPU_MAP")
        slots.update(range(first, last + 1))
    need(len(slots) >= CPUS and memory_mib(mem) >= MEMORY, "EXACT_PARENT_NODE_CPU_MEMORY")


def memory_mib(raw):
    match = re.fullmatch(r"([0-9]+)([MGT]?)", raw)
    need(match is not None, "EXACT_MEMORY_UNITS")
    return int(match[1]) * {"": 1, "M": 1, "G": 1024, "T": 1048576}[match[2]]


def absence(claim, label, query_job, uid):
    parent, node = claim["parent"], claim["node"]
    job = fields(one(query_job("scontrol", "show", "job", parent, "-o")))
    nodes = hosts(job["NodeList"], query_job)
    need(job["JobId"] == parent and job["JobState"] == "RUNNING" and job["NumNodes"] == "40"
         and len(nodes) == 40 and node in nodes and "gres/dcu=320" in job["AllocTRES"].split(",")
         and job["UserId"].endswith(f"({uid})"), "EXACT_OWNED_ACTIVE_PARENT")
    queue_parent, queue_state, queue_list = one(query_job("squeue", "-h", "-j", parent, "-o", "%i|%T|%N")).split("|")
    need(queue_parent == parent and queue_state == "RUNNING" and set(nodes) == set(hosts(queue_list, query_job)),
         "EXACT_SQUEUE_PARENT_AND_NODES")
    machine = fields(one(query_job("scontrol", "show", "node", node, "-o")))
    need(machine["NodeName"] == node and not any(x in machine["State"] for x in ("DOWN", "DRAIN", "FAIL"))
         and int(machine["CPUTot"]) >= CPUS and int(machine["RealMemory"]) >= MEMORY, "EXACT_TARGET_CPU_CAPACITY")
    steps = query_job("scontrol", "show", "step", parent, "-o")
    need(steps, "EXACT_ACTIVE_STEP_OBSERVATION")
    for raw in steps:
        step = fields(raw)
        if step.get("State") not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}:
            check_step(parent, node, step["StepId"], step["NodeList"], query_job)
    for row in query_job("squeue", "--steps", "-h", "-j", parent, "-o", "%i|%N"):
        step_id, node_list = row.split("|")
        check_step(parent, node, step_id, node_list, query_job)
    need(all(name != label for _, name in
             (row.split("|") for row in query_job("sacct", "-nP", "-j", parent, "-o", "JobIDRaw,JobName%100"))),
         "REQUEST_LABEL_ALREADY_OBSERVED")
    parent_resources(parent, node, query_job)
    return utc_epoch(job["EndTime"])


def check_step(parent, node, step_id, node_list, query_job):
    need(pattern(step_id, parent + r"\.(?:[0-9]+|batch|extern)"), "EXACT_PARENT_STEP_ID")
    if step_id.rsplit(".", 1)[1] not in {"batch", "extern"}:
        need(node not in hosts(node_list, query_job), "PLACEMENT_ALREADY_OCCUPIED_BY_NUMERIC_STEP")


def runtime(root, data):
    results = root / "results"
    model = results / "training-qwen-original-public-model-v1/model"
    framework = results / "training-qwen-a68952c-swift-initial-CPU-v1/framework"
    sandbox = root / "containers/dtk26-sandbox-r1"
    hyhal = Path("/usr/local/hyhal")
    binds = ((data, data), (model, model), (framework, framework),
             (results / "training-vlm-swift-offline-runtime-v1", "/swift-runtime"),
             (results / "training-qwen-vision014-CPU-v2/vision", "/vision"),
             (results / "training-minicpm-hygon-processor-r2/cpu/runtime-site", "/minicpm-transformers"),
             (hyhal, "/opt/hyhal"))
    need(APPTAINER.is_file() and all(directory(path) for path, _ in binds) and directory(sandbox)
         and Path("/opt/hyhal").resolve(strict=True) == hyhal, "ORIGINAL_RUNTIME_PATHS")
    read(model / "config.json", 131072, expected=CONFIG_SHA)
    return model, framework, sandbox, binds


def command(claim, label, data, provider):
    model, framework, sandbox, binds = provider
    submit = ["srun", "--jobid=" + claim["parent"], "--job-name=" + label, "--exact", "--nodes=1", "--ntasks=1",
              "--nodelist=" + claim["node"], f"--cpus-per-task={CPUS}", f"--mem={MEMORY}M", "--gres=none",
              "--immediate=5", "--time=7", str(APPTAINER), "exec", "--contain", "--no-home", "--pwd", "/"]
    for host, target in binds:
        submit.extend(("--bind", f"{host}:{target}:ro"))
    process = ["python3", "-B", "-u", "-c", WORKER, claim["parent"], claim["node"],
               str(data / "src/check_textvqa_street_scene_native.py"), "--dataset", str(data / "TRAIN_SWIFT.jsonl"),
               "--inventory", str(data / "MEDIA_PINS.json"), "--inventory-sha", MEDIA_SHA, "--model-dir", str(model)]
    shell = ("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
             "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
             "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; "
             f"export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:{shlex.quote(str(framework))}; "
             "exec /usr/bin/timeout --kill-after=10s 390s " + shlex.join(process))
    return [*submit, str(sandbox), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def start(root, request, publication, claim_file, claim_sha, authority_file, authority_sha, *,
          query_job=query, get_provider=runtime, execute=subprocess.run, clock=time.time, uid=None):
    run = namespace(root / "results", request)
    need(not (run / "intent.json").exists() and not (run / "intent.json").is_symlink(), "OWNER_ATTEMPT_ALREADY_USED")
    claim = json.loads(read(claim_file, 4096, expected=claim_sha))
    authority = json.loads(read(authority_file, 4096, expected=authority_sha))
    placement(claim, request, publication)
    fresh(authority, claim, clock())
    installed(run, request)
    data = run / "inputs"
    argv = command(claim, run.name, data, get_provider(root, data))
    parent_end = absence(claim, run.name, query_job, os.getuid() if uid is None else uid)
    epoch = clock()
    fresh(authority, claim, epoch)
    need(parent_end >= epoch + LIMIT, "LIVE_PARENT_REMAINING_BUDGET")
    intent = dict(request=request, publication=publication, parent=claim["parent"], node=claim["node"],
                  epoch=epoch, deadline=epoch+LIMIT, claim_SHA256=claim_sha, admission_SHA256=authority_sha,
                  argv_SHA256=digest(encode(argv)), planned=ROWS, allocated_DCU=0, replay_allowed=False)
    save(run / "intent.json", encode(intent))
    try:
        with (run / "worker.out").open("xb") as out, (run / "worker.err").open("xb") as err:
            code = execute(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=LIMIT-8).returncode
    except (OSError, subprocess.SubprocessError):
        code = None
    report = dict(request=request, state="READ_INDEPENDENT_TERMINAL", launch_returncode=code,
                  planned=ROWS, replay_allowed=False)
    save(run / "launcher-return.json", encode(report))
    return report


def zero_gres(tres):
    return all(not key.lower().startswith("gres/") or val == "0" for key, val in tres.items())


def utc_epoch(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def terminal(results, request, query_job=query):
    run = namespace(results, request)
    intent = json.loads(read(run / "intent.json", 4096))
    need(intent["request"] == request and intent["replay_allowed"] is False and intent["planned"] == ROWS,
         "EXACT_INDEPENDENT_TERMINAL_INTENT")
    parent, node = intent["parent"], intent["node"]
    rows = [row.split("|") for row in query_job("sacct", "-nP", "-j", parent, "-o", ACCOUNTING)]
    actual = [row for row in rows if len(row) > 1 and row[1] == run.name]
    report = dict(request=request, planned=ROWS, processed=0, native_pass=False, replay_allowed=False,
                  state="ACCOUNTING_UNKNOWN", actual_step=None, allocated_DCU=None, optimizer_updates=0)
    if not actual:
        return report
    jobid, label, state, code, elapsed, placement_node, tres_raw, start_at, end_at = one(actual)
    parts = tres_raw.split(",")
    tres = dict(part.split("=", 1) for part in parts)
    need(pattern(jobid, parent + r"\.[0-9]+") and placement_node == node and len(tres) == len(parts)
         and zero_gres(tres) and tres["node"] == "1" and tres["cpu"] == str(CPUS)
         and memory_mib(tres["mem"]) == MEMORY, "EXACT_ZERO_GRES_NUMERIC_TERMINAL")
    report.update(actual_step=jobid, allocated_DCU=0, scheduler_state=state, scheduler_exit_code=code,
                  accounting_row_SHA256=digest(encode(actual[0])))
    if state not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}:
        return report
    need(0 <= int(elapsed) <= LIMIT and intent["epoch"]-1 <= utc_epoch(start_at) <= utc_epoch(end_at)
         <= intent["deadline"]+1, "EXACT_TERMINAL_BOUNDED_EPOCH")
    report["state"] = "ZERO_GRES_TERMINAL_NATIVE_INCOMPLETE"
    if not regular(run / "worker.out") or not (run / "worker.out").stat().st_size:
        return report
    raw = read(run / "worker.out", 262144)
    lines = raw.splitlines()
    if lines.count(("EXACT_STEP=" + jobid).encode()) != 1:
        return report
    native = []
    for line in lines:
        if b'"collator_passed":' not in line or not line.startswith(b"{"):
            continue
        native.append((line, json.loads(line)))
    if len(native) != 1:
        return report
    evidence_raw, evidence = native[0]
    need(evidence["planned"] == ROWS and 0 <= evidence["processed"] <= ROWS
         and evidence["processed"] == len(evidence["rows"])
         and evidence["dataset_SHA256"] == DATA_SHA and evidence["media_inventory_SHA256"] == MEDIA_SHA
         and evidence["template_SHA256"] == TEMPLATE_SHA and evidence["GPU"] == evidence["optimizer_updates"] == 0
         and evidence["provider_versions"] == VERSIONS
         and evidence["model_materialized"] is False, "EXACT_PRIVATE_NATIVE_REPORT_IDENTITY")
    report.update(processed=evidence["processed"], native_report_SHA256=digest(evidence_raw),
                  failures=len(evidence["failures"]))
    if (state, code, evidence["processed"], len(evidence["failures"]), evidence["collator_passed"]) == (
            "COMPLETED", "0:0", ROWS, 0, True):
        report.update(state="ZERO_GRES_AND_NATIVE_CHECK_COMPLETE", native_pass=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("stage", "start", "terminal"):
        params = modes.add_parser(mode)
        params.add_argument("--request", required=True)
        if mode == "stage":
            params.add_argument("--zip", type=Path, required=True)
            params.add_argument("--receipt", type=Path, required=True)
        if mode == "start":
            params.add_argument("--publication", required=True)
            for name in ("claim", "authority"):
                params.add_argument("--" + name, type=Path, required=True)
                params.add_argument("--" + name + "-sha", required=True)
    args = parser.parse_args()
    need(sys.platform == "linux", "ACTUAL_TARGET_SHARED_HOST_ONLY")
    os.umask(0o077)
    if args.mode == "stage":
        result = stage(ROOT / "results", args.request, args.zip, args.receipt)
    elif args.mode == "start":
        need(all(pattern(sha, "[0-9a-f]{64}") for sha in (args.claim_sha, args.authority_sha)), "EXACT_PIN_FORMAT")
        result = start(ROOT, args.request, args.publication, args.claim, args.claim_sha, args.authority, args.authority_sha)
    else:
        result = terminal(ROOT / "results", args.request)
    print(encode(result).decode().strip())


if __name__ == "__main__":
    main()
