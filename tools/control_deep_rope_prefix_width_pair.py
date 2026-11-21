"""Admit one paired DeepSeek prefix compile and bind the original Slurm result."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from run_deep_rope_prefix_width_compile_pair import PAIR, VARIANTS, host_step_command, read, require


REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT, NODE, ZONE = "1103606", "f12r2n12", "O01"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
CLAIM = "coordination/tasks/deep-inverse-output-rope-prefix-width-pair-cpu-v1.json"
TEMPLATE = "tools/control_deep_inverse_rope_compile.py"
TEMPLATE_SHA = "05e1e3258fab050902650bd9ffc0e71ff9a99badf3dd4af7d6cb0e2ad1980489"
RESULT = ROOT / "results" / (PAIR + "-controller")
HOST_FILES = {
    TEMPLATE: TEMPLATE_SHA,
    "tools/read_training_slurm_membership.py": "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    "coordination/training/training80-active-node-task-map-v1.json": "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    "code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py":
        "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
}
WORKER = "tools/run_deep_rope_prefix_width_compile_pair.py"
SOURCE = "code/bw1000/Deep-C4-causal-consumer-r483"
MANIFESTS = {arm.width: f"{SOURCE}/inverse-output-rope-prefix-pack{arm.width}-target-v1/SOURCE_FILES.json"
             for arm in VARIANTS}
WORKLOAD = sha256("\n".join(arm.manifest for arm in VARIANTS).encode()).hexdigest()
REQUEST = sha256(f"{PAIR}:{PARENT}:{NODE}:{WORKLOAD}".encode()).hexdigest()[:32]


def check_scope(repo, claim_sha):
    claim = json.loads(read(repo / CLAIM, 8192, claim_sha))
    require(claim["request"] == REQUEST and claim["label"] == PAIR and claim["placement"] == PLACEMENT
            and claim["planned_device_TUs"] == 2 and claim["wall_seconds"] == 300
            and claim["workload_SHA256"] == WORKLOAD
            and claim["CPU_GO"] is True and claim["GPU_GO"] is claim["replay_allowed"] is False,
            "paired compile scope differs")
    files = claim["host_files"]
    require(files.keys() == {*HOST_FILES, WORKER, str(Path(__file__).relative_to(REPO).as_posix())}
            and all(files[key] == pin for key, pin in HOST_FILES.items()), "unexpected pair HOST source set")
    names = {CLAIM, *files}
    for name, pin in files.items():
        read(repo / name, 2 * 1024**2, pin)
    for arm in VARIANTS:
        relative = MANIFESTS[arm.width]
        path = repo / relative
        names.add(relative)
        manifest = json.loads(read(path, 8192, arm.manifest))["files_SHA256"]
        require(manifest.keys() == {"PLAN.json", "compile_once.py", "inverse_output_rope.cu"}, "variant package members differ")
        for name, pin in manifest.items():
            member = path.with_name(name)
            read(member, 128 * 1024, pin)
            names.add(member.relative_to(repo).as_posix())
    return names


def check_result(output, row, epoch):
    if not row or row[2:4] != ["COMPLETED", "0:0"]:
        return None
    from read_training_slurm_membership import allocated_dcu

    resources = dict(field.split("=", 1) for field in row[6].split(","))
    require(re.fullmatch(PARENT + r"\.[0-9]+", row[0]) and row[1] == PAIR and row[5] == NODE
            and row[4].isdecimal() and int(row[4]) <= 300
            and resources.get("cpu") == "4" and resources.get("node") == "1"
            and resources.get("mem") in ("8192M", "8G") and allocated_dcu(row[6]) == 0,
            "paired original accounting differs")
    raw = read(output / "CPU-pair-result.json", 64 * 1024)
    result = json.loads(raw)
    expected = dict(label=PAIR, original_step=row[0], node=NODE, epoch=epoch, deadline=epoch+300,
                    planned_device_TUs=2, verified_device_TUs=2, actual_GPU_steps=0, completed=True,
                    passed=True, device_qualified=False)
    require(all(type(result.get(key)) is type(value) and result[key] == value for key, value in expected.items()),
            "paired compiler receipt is incomplete or different")
    objects = {}
    for arm, evidence in zip(VARIANTS, result["variant_results"], strict=True):
        base = output.parent / arm.label
        require(evidence["width"] == arm.width and evidence["manifest_SHA256"] == arm.manifest
                and evidence["invoked"] is evidence["verified"] is True and evidence["returncode"] == 0,
                "paired arm ordinal or compile status differs")
        worker_raw = read(base / "CPU-result.json", 1024 * 1024, evidence["worker_receipt_SHA256"])
        worker = json.loads(worker_raw)
        require(worker["label"] == arm.label and worker["original_step"] == row[0] and worker["node"] == NODE
                and worker["epoch"] == epoch and worker["deadline"] == epoch+300
                and worker["manifest_SHA256"] == arm.manifest and worker["passed"] is True
                and worker["completed"] is True and worker["exports"] == [arm.export] and worker["actual_device_TUs"] == 1
                and worker["actual_GPU_steps"] == worker["explicit_Runtime_calls"] == 0
                and worker["object"] == f"/refactor/candidates/{arm.label}/inverse_output_rope.o",
                "target arm native compiler receipt differs")
        for file, key, ceiling in (
            ("inverse_output_rope.o", "object_SHA256", 16 * 1024**2),
            ("inverse_output_rope.before.d", "fresh_M_SHA256", 2 * 1024**2),
            ("inverse_output_rope.d", "actual_MD_SHA256", 2 * 1024**2),
        ):
            require(worker[key] == evidence[key], "paired result and original worker digest differ")
            read(base / file, ceiling, evidence[key])
        objects[f"pack{arm.width}"] = evidence["object_SHA256"]
    return dict(receipt_SHA256=sha256(raw).hexdigest(), object_SHA256=objects)


def prelaunch():
    require(all(not (ROOT / "candidates" / arm.label).exists() and
                not (ROOT / "candidates" / arm.label).is_symlink() for arm in VARIANTS),
            "original variant scope exists; reconcile before any new command")


def reconcile(run_path, output, claim_sha, query, terminal):
    raw = read(run_path / "intent.json", 64 * 1024)
    intent = json.loads(raw)
    epoch = intent["epoch"]
    require(intent["request"] == REQUEST and intent["parent"] == PARENT and intent["node"] == NODE
            and intent["claim_SHA256"] == claim_sha and intent["source_manifest_SHA256"] == WORKLOAD
            and intent["replay_allowed"] is False and type(epoch) is float and epoch > 0
            and intent["deadline"] == epoch+300, "original paired intent differs")
    row = query(8)
    result = dict(request=REQUEST, state="UNKNOWN", row=row, intent_SHA256=sha256(raw).hexdigest(),
                  independent_observation=True, replay_allowed=False)
    if terminal(row):
        result["state"] = "TERMINAL_UNVERIFIED"
        try:
            verified = check_result(output, row, epoch)
            if verified is not None:
                result.update(verified, state="COMPILE_VERIFIED")
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["error"] = type(error).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--reconcile-existing", action="store_true")
    args = parser.parse_args()
    check_scope(REPO, args.claim_sha256)
    scope = {"__name__": "frozen_deep_cpu_controller", "__file__": str(REPO / TEMPLATE)}
    source = read(REPO / TEMPLATE, 64 * 1024, TEMPLATE_SHA)
    exec(compile(source, scope["__file__"], "exec"), scope)
    scope.update(LABEL=PAIR, REQUEST=REQUEST, MANIFEST=WORKLOAD, check_result=check_result,
                 command=lambda epoch: host_step_command(REPO.as_posix(), PARENT, NODE, epoch))
    helper = scope["ACCOUNTING"].as_posix()
    accounting = {"__name__": "frozen_slurm_accounting"}
    exec(compile(read(REPO / helper, 8192, HOST_FILES[helper]), helper, "exec"), accounting)
    output = ROOT / "candidates" / PAIR
    if args.reconcile_existing:
        result = reconcile(RESULT, output, args.claim_sha256, scope["accounting"], accounting["is_terminal"])
    else:
        prelaunch()
        result = scope["controller"](RESULT, output, args.claim_sha256, accounting["wait_terminal"])
    print(json.dumps({key: result.get(key) for key in ("request", "state", "row", "object_SHA256", "receipt_SHA256",
                                                       "intent_SHA256", "independent_observation", "error")},
                     allow_nan=False), flush=True)
    return int(result["state"] != "COMPILE_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
