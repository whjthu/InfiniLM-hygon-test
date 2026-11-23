"""Verify original six-rank terminal evidence from Slurm, ranks, and provider traces."""

from hashlib import sha256
import json
from pathlib import Path
import re
import statistics
import subprocess


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/o01-six-rank-restored-fleet-shca-v1")
REQUEST = "37b242d312124f7682326f2151d2c8d4"
PUBLISHED = "1988f356b99708106a63bfd8ecbe949111d6239e"
MANIFEST = "831238e565e0777eae3d14c4ede3fcfecf8f6e77fb994adc847c75a57b1a0a47"
COLUMNS = ("JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
CONTROLLER_COLUMNS = ("JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES")
PHASES = ("pre", "gpu", "post")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "REVOKED"}
FAILURE_MARKERS = {
    "network_unreachable": rb"network is unreachable|no route to host",
    "connection_refused": rb"connection refused",
    "rdma_completion_error": rb"completion.*(?:error|status)|transport retry counter exceeded",
    "rdma_connection_manager_error": rb"rdmacm.*(?:error|fail)|ibv_(?:create|modify|reg).*fail",
    "address_error": rb"cannot assign requested address|invalid address",
    "collective_timeout": rb"timed out|watchdog.*timeout|connection timed out",
    "internal_or_system_error": rb"ncclInternalError|ncclSystemError|unhandled system error",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(name, limit=1_048_576):
    path = ROOT/name
    require(path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit, "Invalid scoped file: "+name)
    return path.read_bytes()


def main():
    manifest = read("src/SOURCE_FILES.json")
    require(sha256(manifest).hexdigest() == MANIFEST, "Unexpected frozen package")
    source = json.loads(manifest)["files"]
    for name, pin in source.items():
        raw = read("src/"+name)
        require(len(raw) == pin["bytes"] and sha256(raw).hexdigest() == pin["SHA256"], "Original source drift")
    claim = json.loads(read("src/CLAIM.json"))
    places = claim["placements"]
    expect = {f"{ROOT.name}-{phase}-r{i}": (parent, node, phase)
              for i, (parent, node) in enumerate(places) for phase in PHASES}
    raw = read("outcome.json")
    outcome = json.loads(raw)
    intent = json.loads(read("intent.json"))
    require(intent["claim"] == outcome["claim"] == PUBLISHED and intent["request_id"] == outcome["request_id"] == REQUEST
            and intent["nodes"] == [node for _, node in places]
            and outcome["accounting_columns"] == list(CONTROLLER_COLUMNS),
            "Original source, request or accounting schema mismatch")
    accounting = {}
    for parent in sorted({parent for parent, _ in places}):
        calls = subprocess.check_output(["sacct","-n","-P","-j",parent,"--format="+",".join(COLUMNS)],
                                       text=True, timeout=10)
        for line in calls.splitlines():
            row = line.split("|")
            if len(row) != len(COLUMNS) or row[1] not in expect:
                continue
            p, n, phase = expect[row[1]]
            require(parent == p and row[5] == n and row[1] not in accounting and row[2].split()[0] in TERMINAL
                    and re.fullmatch(re.escape(parent)+r"\.\d+", row[0]), "Original step binding mismatch")
            require(("gres/dcu=1" in row[6].split(",")) == (phase == "gpu"), "Assigned GPU differs from scope")
            accounting[row[1]] = row
    require(sorted(accounting.values()) == sorted(outcome["accounting"]), "Step denominator drift")
    result = dict(request=REQUEST, publication=PUBLISHED, manifest_SHA256=MANIFEST, planned_ranks=6,
                  verified_ranks=0, planned_numeric_rank_checks=450, verified_numeric_rank_checks=0,
                  actual_accounting=accounting, accounting_columns=COLUMNS, passed=outcome["passed"],
                  outcome_SHA256=sha256(raw).hexdigest(), wall_seconds=outcome["wall_seconds"],
                  independent_post=False, measured_gpu_device_busy=False, formal_training=False, formal_model=False)
    for phase in PHASES:
        observed = outcome.get(phase)
        if observed is None:
            continue
        require(json.loads(read(phase+"-receipt.json")) == observed, "Phase receipt differs")
        for row in observed["steps"]:
            name = row["name"]
            log = ROOT/(name+".log")
            require(log.stat().st_size <= 1_048_576 and sha256(log.read_bytes()).hexdigest() == row["log_SHA256"],
                    "Per-rank log changed")
    gpu, post = outcome.get("gpu"), outcome.get("post")
    result["actual_GPU_steps"] = 0 if gpu is None else sum(row["name"] in accounting for row in gpu["steps"])
    result["controller_forced_GPU_subprocess_cleanup"] = bool(gpu and gpu["forced"])
    result["failure_trace"] = []
    for i, (_, node) in enumerate(places):
        trace_name = f"traces/{i}-failure.json"
        if (ROOT/trace_name).is_file():
            raw_trace = read(trace_name)
            failure = json.loads(raw_trace)
            name = f"{ROOT.name}-gpu-r{i}"
            require(name in accounting and failure["rank"] == i and failure["node"] == node
                    and failure["step"] == accounting[name][0], "Failure trace binds to a different job")
            log = read(name+".log")
            tags = {tag:len(re.findall(regex, log, flags=re.IGNORECASE)) for tag, regex in FAILURE_MARKERS.items()}
            result["failure_trace"].append(dict(rank=i, failed_stage=failure["failed_stage"],
                                                error_type=failure["error_type"], markers=tags,
                                                SHA256=sha256(raw_trace).hexdigest()))
    if post is not None and post["passed"]:
        checked_posts = 0
        for i, (_, node) in enumerate(places):
            before, after = (json.loads(read(f"{arm}/{node}.json")) for arm in ("pre","post"))
            require(before["boot_id"] == after["boot_id"] and before["pci"] == after["pci"] == claim["pcis"][node]
                    and before["kfd_holder_count"] == after["kfd_holder_count"] == 0
                    and after["step"] == accounting[f"{ROOT.name}-post-r{i}"][0],
                    "Original post did not independently establish GPU release")
            checked_posts += 1
        result["independent_post"] = checked_posts == 6
    if not result["passed"]:
        result["verified_rank_records_when_failed"] = [x["rank"] for x in outcome["ranks"]]
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    require(len(accounting) == 18 and all(row[2:4] == ["COMPLETED","0:0"] for row in accounting.values()),
            "Successful run lacks complete step accounting")
    providers = json.loads(read("src/PROVIDERS.json"))["files"]
    ranks, transport = [], []
    for i, (parent, node) in enumerate(places):
        row = json.loads(read(f"ranks/{i}.json"))
        handle = accounting[f"{ROOT.name}-gpu-r{i}"][0]
        require((row["rank"],row["node"],row["step"]) == (i,node,handle) and row in outcome["ranks"]
                and row["backend"] == "nccl" and row["mapped_provider_pins"] == providers and row["kfd"] > 0,
                "Assigned rank, KFD or transport provider differs")
        for phase, folder in (("pre","pre"),("gpu","gpu-gate"),("post","post")):
            snap = json.loads(read(f"{folder}/{node}.json"))
            require(snap["step"] == accounting[f"{ROOT.name}-{phase}-r{i}"][0]
                    and snap["boot_id"] == row["boot_id"] and snap["pci"] == claim["pcis"][node]
                    and row["pci"] in snap["pci"] and snap["kfd_holder_count"] == 0,
                    "Rank device ownership or post differs")
        trace = f"traces/{i}-first-allreduce-returned.json"
        first = read(trace)
        check = json.loads(first)
        require(check["step"] == handle and check["node"] == node
                and all(check["providers"].get(k) == v for k,v in providers.items()), "Live provider differs")
        log = read(f"{ROOT.name}-gpu-r{i}.log")
        network = re.findall(rb"Using network\s+([\w.-]+)", log)
        shca = len(re.findall(rb"via NET/IBext(?:_v8)?(?:/|\b)", log))
        socket = len(re.findall(rb"via NET/Socket(?:/|\b)", log))
        require(network == [b"IBext_v8"] and shca > 0 and socket == 0, "Actual transport is unqualified")
        transport.append(dict(rank=i,parent=parent,node=node,physical_PCI=row["pci"],SHCA_channels=shca,
                              socket_channels=socket,first_return_trace_SHA256=sha256(first).hexdigest()))
        ranks.append(row)
    require(len({(r["node"],r["pci"],r["kfd"]) for r in ranks}) == 6, "Duplicate assigned device")
    timings = []
    for payload, *rows in zip(claim["rank_payload_bytes"],*(r["rows"] for r in ranks),strict=True):
        require(all(r["bytes"] == payload and r["checked"] == 25 and len(r["latency_ns"]) == 20
                    and all(isinstance(t,int) and t > 0 for t in r["latency_ns"]) for r in rows),
                "Incomplete numerical or measured denominator")
        slowest = sorted(max(sample) for sample in zip(*(r["latency_ns"] for r in rows),strict=True))
        timings.append(dict(payload_bytes=payload,paired_slowest_rank_measurements=len(slowest),
                            median_slowest_rank_us=statistics.median(slowest)/1_000,
                            p95_slowest_rank_us=slowest[18]/1_000))
    result.update(verified_ranks=len(ranks),verified_numeric_rank_checks=450,
                  transport=transport,timing=timings)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
