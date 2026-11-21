"""Launch or reconcile the single admitted GLM fresh-KV compile step."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v1"
PARENT, NODE, ZONE = "1103606", "f12r2n13", "O01"
EXPORTS = ["glm_r474_fresh_kv_pack_t128", "glm_r474_fresh_kv_pack_t64"]
SOURCE = HERE.parent / "fresh-kv-pack-CPU-v1"
SOURCE_PUBLICATION = "47f8380857d2ec58e549219a09bccafa26211299"
SOURCE_MANIFEST = "a5e43581817e7fad431440bfa79ff3eba3708dec095dc05b76675b6d7388d259"
SOURCE_CLAIM = Path("coordination/tasks/glm-r474-fresh-kv-pack-gfx936-cpu-v1.json")
SOURCE_CLAIM_PIN = "6c20deb38e5a726501f044588afc24047a07fd76e3701fe9f2104694f9da2723"
MANIFEST = "333551bc79cbd2e56e18274737b00c57c0a918cc00700b167a61b9a316ac7850"
COMPILER = "d598ee49072a880a0c2a07f1cc7043e8e4d28f406bfe9a5e1352bb14e8723280"
CONTROLLER = Path("tools/control_deep_inverse_rope_compile.py")
MEMBERSHIP = Path("tools/read_training_slurm_membership.py")
LEDGER = Path("coordination/training/training80-active-node-task-map-v1.json")
ACCOUNTING = Path("code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py")
PINS = {
    CONTROLLER: "05e1e3258fab050902650bd9ffc0e71ff9a99badf3dd4af7d6cb0e2ad1980489",
    MEMBERSHIP: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    LEDGER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    ACCOUNTING: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
}
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
REQUEST = sha256(f"{LABEL}:{PARENT}:{NODE}:{MANIFEST}".encode()).hexdigest()[:32]
SOURCE_LEAF = f"/refactor/incoming/{LABEL}/fresh_kv_pack.cu"
OBJECT = f"/refactor/candidates/{LABEL}/fresh_kv_pack.o"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected=None, limit=2**21, allow_empty=False):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    require((raw or allow_empty) and len(raw) <= limit, f"bounded artifact missing or too large: {path.name}")
    require(expected is None or sha256(raw).hexdigest() == expected, f"frozen SHA differs: {path.name}")
    return raw


def claim_identity(claim_sha, stage, *, reconcile=False):
    require(re.fullmatch("[0-9a-f]{64}", claim_sha), "exact proposed HOST claim SHA required")
    claim = json.loads(pin(HERE / "CLAIM.json", claim_sha))
    require(claim["request"] == REQUEST and claim["placement"] == PLACEMENT and claim["label"] == LABEL
            and claim["wall_seconds"] == 300 and claim["planned_device_translation_units"] == 1
            and claim["published_source_commit"] == SOURCE_PUBLICATION
            and claim["source_manifest_SHA256"] == SOURCE_MANIFEST and claim["target_manifest_SHA256"] == MANIFEST
            and claim["CPU_GO"] is True and claim["GPU_GO"] is False and claim["replay_allowed"] is False,
            "fixed source or CPU admission claim differs")
    require(claim["host_entry_SHA256"] == sha256(pin(Path(__file__))).hexdigest()
            and claim["shared_files"] == {p.as_posix(): h for p, h in PINS.items()}, "invalid HOST source pins")
    source_claim = json.loads(pin(REPO / SOURCE_CLAIM, SOURCE_CLAIM_PIN))
    require(source_claim["label"] == LABEL and source_claim["fixed_package"]["source_manifest_SHA256"] == SOURCE_MANIFEST
            and source_claim["fixed_package"]["generated_target_manifest_SHA256"] == MANIFEST,
            "published upstream scope differs")
    index = json.loads(pin(SOURCE / "SOURCE_FILES.json", SOURCE_MANIFEST))["files_SHA256"]
    for name, digest in index.items():
        require(Path(name).name == name, "unexpected source member name")
        pin(SOURCE / name, digest)
    plan = json.loads(pin(SOURCE / "PLAN.json", index["PLAN.json"]))
    if not reconcile:
        require(stage.resolve(strict=True) == stage and stage.is_dir(), "stage is absent or noncanonical")
        require(not (stage / "SOURCE_FILES.json").is_symlink(), "generated manifest link refused")
        files = json.loads(pin(stage / "SOURCE_FILES.json", MANIFEST))["files_SHA256"]
        require(files == {"PLAN.json": index["PLAN.json"], "fresh_kv_pack.cu": index["fresh_kv_pack.cu"],
                          "compile_once.py": COMPILER}, "unexpected generated stage identity")
        for name, digest in files.items():
            require(not (stage / name).is_symlink(), "generated member link refused")
            pin(stage / name, digest)
    return claim, plan


def dependency_paths(raw):
    target, colon, words = raw.decode("utf-8").replace("\\\n", " ").partition(":")
    paths = set(shlex.split(words))
    require(target.strip() == OBJECT and colon and paths and all(p.startswith("/") for p in paths),
            "original M or MD dependency target differs")
    return paths


def exact_tres(raw):
    fields = [value.split("=", 1) for value in raw.split(",")]
    require(all(len(field) == 2 and all(field) for field in fields), "malformed original CPU TRES")
    parsed = dict(fields)
    require(len(parsed) == len(fields) and parsed.get("cpu") == "4" and parsed.get("node") == "1"
            and parsed.get("mem") in {"8G", "8192M", "8388608K"} and
            all(value.isdecimal() and int(value) == 0 for name, value in parsed.items() if name.startswith("gres/")),
            "actual accounting is outside the zero-GRES slot")


def terminal_pass(path, row, epoch, plan):
    if not row or row[2:4] != ["COMPLETED", "0:0"]:
        return None
    require(re.fullmatch(PARENT + r"\.[0-9]+", row[0]) and row[1] == LABEL and row[5] == NODE
            and row[4].isdecimal() and int(row[4]) <= 300, "foreign terminal Slurm step")
    exact_tres(row[6])
    content = pin(path / "CPU-result.json", limit=2**20)
    result = json.loads(content)
    expected = dict(label=LABEL, manifest_SHA256=MANIFEST, original_step=row[0], node=NODE, epoch=epoch,
                    deadline=epoch+300, GRES="none", planned_device_TUs=1, actual_device_TUs=1,
                    actual_GPU_steps=0, explicit_Runtime_calls=0, exports=EXPORTS, object=OBJECT,
                    passed=True, completed=True, device_qualified=False)
    require(all(type(result.get(key)) is type(value) and result[key] == value
                for key, value in expected.items()), "compiler receipt scope differs")
    before = pin(path / "fresh_kv_pack.before.d", result["fresh_M_SHA256"])
    after = pin(path / "fresh_kv_pack.d", result["actual_MD_SHA256"])
    deps = result["dependencies_SHA256"]
    header = plan["cuda_runtime_header"]
    require(dependency_paths(before) == dependency_paths(after) == set(deps)
            and deps.get(SOURCE_LEAF) == plan["source_SHA256"]
            and deps.get(header) == plan["cuda_runtime_header_SHA256"]
            and all(re.fullmatch("[0-9a-f]{64}", h) for h in deps.values()), "original fresh M/actual MD differ")

    flags = plan["compiler_prefix"]
    commands = {
        "fresh-M": flags + ["-M", "-MF", OBJECT.removesuffix(".o") + ".before.d", "-MT", OBJECT, SOURCE_LEAF],
        "compile": flags + ["-MD", "-MF", OBJECT.removesuffix(".o") + ".d", "-MT", OBJECT, "-c", SOURCE_LEAF, "-o", OBJECT],
        "symbols": ["/usr/bin/nm", "-g", "--defined-only", OBJECT],
    }
    require([cmd["label"] for cmd in result["commands"]] == list(commands), "wrong original compiler command order")
    logs = {}
    for cmd in result["commands"]:
        label = cmd["label"]
        require(cmd["returncode"] == 0 and cmd["argv"] == commands[label], "wrong original gfx936 command")
        for stream in ("stdout", "stderr"):
            logs[label, stream] = pin(path / f"{label}.{stream}", cmd[stream + "_SHA256"], allow_empty=True)
    named = {parts[-1] for line in logs["symbols", "stdout"].decode().splitlines()
             if len(parts := line.split()) >= 3 and parts[-2] == "T" and parts[-1].startswith("glm_r474_fresh_kv_pack_")}
    require(named == set(EXPORTS), "wrong original object strong symbols")
    obj = pin(path / "fresh_kv_pack.o", result["object_SHA256"], limit=16*1024**2)
    require(len(obj) >= 64 and obj[:7] == b"\x7fELF\x02\x01\x01"
            and struct.unpack_from("<HHI", obj, 16) == (1, 62, 1), "original compiler object lacks host ELF64 REL")
    return dict(receipt_SHA256=sha256(content).hexdigest(), object_SHA256=sha256(obj).hexdigest(),
                fresh_M_SHA256=sha256(before).hexdigest(), actual_MD_SHA256=sha256(after).hexdigest(),
                terminal_row_SHA256=sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest(),
                planned_device_TUs=1, actual_device_TUs=1)


def shared_controller(stage, plan):
    sources = {p: pin(REPO / p, digest) for p, digest in PINS.items()}
    membership = ModuleType("read_training_slurm_membership")
    exec(compile(sources[MEMBERSHIP], str(REPO / MEMBERSHIP), "exec"), membership.__dict__)
    scope = {"__name__": "frozen_deep_controller", "__file__": str(REPO / CONTROLLER)}
    with patch.dict(sys.modules, {"read_training_slurm_membership": membership}):
        exec(compile(sources[CONTROLLER], scope["__file__"], "exec"), scope)
    scope.update(REPO=REPO, ROOT=ROOT, LABEL=LABEL, PARENT=PARENT, NODE=NODE, ZONE=ZONE,
                 PACKAGE=stage, MANIFEST=MANIFEST, REQUEST=REQUEST, EXPORTS=EXPORTS,
                 check_result=lambda path, row, epoch: terminal_pass(path, row, epoch, plan))
    waiter = {"__name__": "frozen_terminal_accounting"}
    exec(compile(sources[ACCOUNTING], str(REPO / ACCOUNTING), "exec"), waiter)
    return scope, waiter


def account_both(scope, timeout):
    raw = subprocess.check_output(["sacct", "-n", "-P", "-j", "1087656,1103606",
                                   "-S", "2026-11-20T00:00:00", "-o", ",".join(COLUMNS)], timeout=timeout)
    return scope["accounting_row"](raw)


def inspect_original(run_path, output, claim_sha, query, is_terminal, plan):
    content = pin(run_path / "intent.json")
    intent = json.loads(content)
    epoch = intent["epoch"]
    require(intent["request"] == REQUEST and intent["claim_SHA256"] == claim_sha
            and intent["source_manifest_SHA256"] == MANIFEST and intent["parent"] == PARENT and intent["node"] == NODE
            and intent["replay_allowed"] is False and type(epoch) is float and math.isfinite(epoch)
            and epoch > 0 and intent["deadline"] == epoch + 300, "not the original one-shot request")
    row = query(8)
    prior_path = run_path / "accounting-result.json"
    prior = json.loads(pin(prior_path)) if prior_path.exists() else {}
    if prior:
        require(prior["request"] == REQUEST and prior["accounting_columns"] == COLUMNS
                and prior["actual_GPU_steps"] == 0, "original controller receipt identity differs")
    if prior.get("terminal_visible") is True:
        require(row == prior["row"], "original controller and current accounting disagree")
    evidence = dict(request=REQUEST, state="UNKNOWN", row=row, actual_GPU_steps=0,
                    intent_SHA256=sha256(content).hexdigest(), independent_observation=True)
    if is_terminal(row):
        evidence["state"] = "TERMINAL_UNVERIFIED"
        valid = terminal_pass(output, row, epoch, plan)
        if valid:
            if prior.get("state") == "COMPILE_VERIFIED":
                require(all(prior.get(key) == value for key, value in valid.items()), "terminal artifact SHA changed")
            evidence.update(valid, state="COMPILE_VERIFIED")
    return evidence


def execute(claim_sha, *, reconcile=False, query=None, observe_slot=None, runner=None, waiter_override=None):
    stage = ROOT / "incoming" / (LABEL + "-host-source")
    _, plan = claim_identity(claim_sha, stage, reconcile=reconcile)
    scope, waiter = shared_controller(stage, plan)
    run_path, output = ROOT / "results" / (LABEL + "-controller"), ROOT / "candidates" / LABEL
    lookup = query if query is not None else lambda timeout: account_both(scope, timeout)
    if reconcile:
        return inspect_original(run_path, output, claim_sha, lookup, waiter["is_terminal"], plan)
    return scope["controller"](run_path, output, claim_sha, waiter_override or waiter["wait_terminal"],
                               query=lookup, observe_slot=observe_slot or scope["check_slot"],
                               run=runner or subprocess.run)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--reconcile", action="store_true", help="read existing intent and one accounting observation")
    options = parser.parse_args(args)
    result = execute(options.claim_sha256, reconcile=options.reconcile)
    print(json.dumps(result, allow_nan=False), flush=True)
    return int(result["state"] != "COMPILE_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
