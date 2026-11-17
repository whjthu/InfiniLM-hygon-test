"""Independently read the one original zero-GPU snapshot-I/O gate; never submit work."""
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL = "training-qwen-treasury-snapshot-io-v1"
R = ROOT / LABEL
REQUEST = "63c2aad05ff6451495951f5686141a23"
PUBLICATION = "043a1d4b59661d621ec329b8bbfbbc3c06a77948"
CLAIM_SHA = "e4e1d607d414f9a93bf6acd0d66155e25c84242ce37a577334a31b12c251a4d0"
MF_SHA = "dd37b7f90095eff11a6f08c85ce52d34752d6b758a03c111422493f4cf6eaaa8"
SPEC = {"base": ("1087656", "a01r4n14"), "old_domain": ("1087656", "a01r4n16"), "domain": ("1103606", "f13r1n10")}
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
PINS = {}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def bad_constant(_):
    raise ValueError("non-finite JSON constant")


def decode(raw):
    return json.loads(raw.decode("utf8"), object_pairs_hook=unique_pairs, parse_constant=bad_constant)


def raw(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= 1024**2, "unsafe original small file")
    value = path.read_bytes()
    after = path.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "original changed")
    PINS[str(path)] = hashlib.sha256(value).hexdigest()
    return value


def read(path):
    return decode(raw(path))


def snapshot(raw_bytes, node, step):
    value = decode(raw_bytes)
    fields = {"kind", "mode", "node", "step", "pci", "pci_to_kfd", "kfd_holder_count", "full_fd_inventory_performed", "holder_inventory_basis", "boot_id", "monotonic_ns"}
    require(type(value) is dict and set(value) == fields, "snapshot exact11 schema differs")
    require(value["kind"] == "training_qwen_device_snapshot" and value["mode"] == "host_complete"
            and value["node"] == node and value["step"] == step and re.fullmatch(r"[0-9]+\.[0-9]+", step), "snapshot identity differs")
    require(type(value["kfd_holder_count"]) is int and value["kfd_holder_count"] == 0
            and value["full_fd_inventory_performed"] is True
            and value["holder_inventory_basis"] == "kernel_KFD_plus_sameUID_FD_complete"
            and value["pci_to_kfd"] is None, "HOST fullFD/holder qualification differs")
    pci = value["pci"]
    require(type(pci) is list and len(pci) == 8 and all(type(x) is str for x in pci)
            and len(set(pci)) == 8 and pci == sorted(pci)
            and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in pci), "eight PCI identity differs")
    require(type(value["boot_id"]) is str and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value["boot_id"])
            and type(value["monotonic_ns"]) is int and value["monotonic_ns"] > 0, "boot/clock anchor differs")
    return value


def audit():
    intent, outcome = read(R/"intent.json"), read(R/"outcome.json")
    claim, mf = read(R/"source/CLAIM.json"), read(R/"source/SOURCE_FILES.json")
    require(PINS[str(R/"source/CLAIM.json")] == CLAIM_SHA and PINS[str(R/"source/SOURCE_FILES.json")] == MF_SHA, "published original source differs")
    require(len(mf["files"]) + 1 == 11, "source11 denominator differs")
    for name, info in mf["files"].items():
        require(Path(name).name == name, "unsafe source member")
        b = raw(R/"source"/name)
        require(len(b) == info["bytes"] and hashlib.sha256(b).hexdigest() == info["SHA256"], "source member differs")
    require(intent["request"] == outcome["request"] == REQUEST and intent["publication"] == PUBLICATION
            and intent["claim_SHA256"] == CLAIM_SHA and intent["epoch"] == outcome["epoch"]
            and intent["deadline"] == outcome["deadline"] == intent["epoch"] + 600, "original request/epoch differs")
    require(intent["actual_expected_GPU_cards"] == claim["actual_expected_GPU_cards"] == 0
            and intent["GPU_GO"] is claim["GPU_GO"] is outcome["GPU_GO"] is False
            and outcome["actual_GPU_steps"] == 0 and outcome["automatic_GPU_followon"] is False
            and claim["maximum_CPU_steps"] == 3 and outcome["passed"] is True
            and set(outcome["arms"]) == set(SPEC), "CPU scope or original result differs")
    rows = []
    for parent in ("1087656", "1103606"):
        text = subprocess.check_output(["sacct", "-n", "-P", "-j", parent, "-o", COLS], text=True, timeout=9)
        rows.extend(row for line in text.splitlines() if len(row := line.split("|")) == 7 and row[1] in {LABEL+"-"+a for a in SPEC})
    require(len(rows) == 3, "actual complete3 accounting differs")
    arms = {}
    for arm, (parent, node) in SPEC.items():
        folder = R/"arms"/arm
        a, step_intent = read(folder/"outcome.json"), read(folder/"step-intent.json")
        matches = [row for row in rows if row[1] == LABEL+"-"+arm]
        require(len(matches) == 1, "missing/duplicate actual step")
        row = matches[0]
        require(row == a["accounting"]["row"] == outcome["arms"][arm]["accounting"]["row"]
                and row[0].startswith(parent+".") and row[2:4] == ["COMPLETED", "0:0"]
                and row[5] == node and 0 <= int(row[4]) <= 120, "actual terminal CPU row differs")
        tres = dict(x.split("=",1) for x in row[6].split(",") if "=" in x)
        require(tres == {"cpu":"2","mem":"4G","node":"1"}, "CPU2/mem4G/0GRES actual resource differs")
        require(a == outcome["arms"][arm] and a["passed"] is True and a["rc"] == 0 and a["error"] is None
                and a["actual_GPU_cards"] == 0 and a["accounting_columns"] == COLS
                and step_intent["epoch"] == intent["epoch"] and step_intent["deadline"] == intent["deadline"], "per-arm original outcome differs")
        cmd = step_intent["command"]
        require(cmd[0] == "srun" and "--gres=none" in cmd and "-c2" in cmd and "--mem=4096M" in cmd
                and "--jobid="+parent in cmd and "--job-name="+LABEL+"-"+arm in cmd
                and cmd[cmd.index("-w")+1] == node, "actual CPU command differs")
        limit = step_intent["limits"]
        require(limit["Slurm_seconds"] in (60,120) and limit["inside_seconds"] <= 60
                and limit["inside_seconds"]+5 < limit["Slurm_seconds"]
                and "--time=00:%02d:00" % (limit["Slurm_seconds"]//60) in cmd, "original budget/minute floor differs")
        stdout, stderr = raw(folder/"snapshot.stdout"), raw(folder/"snapshot.stderr")
        value = snapshot(stdout, node, row[0])
        require(value == a["snapshot"] and PINS[str(folder/"snapshot.stdout")] == a["stdout_SHA256"]
                and PINS[str(folder/"snapshot.stderr")] == a["stderr_SHA256"] and len(stderr) == a["stderr_bytes"], "separated raw streams differ")
        lines = stderr.splitlines()
        marker = b"snapshot-io-gate: retained independent stderr marker"
        require(lines.count(marker) == 1, "stderr marker missing/duplicated")
        warnings = [line for line in lines if line.startswith(b"srun: request too frequently, retrying")]
        other = [line for line in lines if line != marker and line not in warnings]
        arms[arm] = dict(actual_accounting=row, snapshot_stdout_SHA256=a["stdout_SHA256"],
            stderr_SHA256=a["stderr_SHA256"], stderr_bytes=len(stderr), stderr_lines=len(lines),
            retained_marker_count=1, srun_rate_warning_count=len(warnings),
            warning_line_SHA256=[hashlib.sha256(line).hexdigest() for line in warnings],
            other_stderr_line_SHA256=[hashlib.sha256(line).hexdigest() for line in other],
            strict_snapshot11_verified=True, actual_PCI_count=8, full_FD_inventory=True, KFD_holders=0,
            node=node, step=row[0], boot_id=value["boot_id"], snapshot_monotonic_ns=value["monotonic_ns"],
            actual_GPU_cards=0)
    return dict(state="ORIGINAL_THREE_HOST_SNAPSHOT_IO_CPU_GATES_INDEPENDENTLY_VERIFIED",request=REQUEST,
        source_publication=PUBLICATION,epoch=intent["epoch"],deadline=intent["deadline"],arms=arms,
        original_artifact_pins=PINS,source_files_verified=11,complete_arm_denominator=3,
        original_CPU_gate_verified=True,actual_GPU_steps=0,GPU_GO=False,automatic_GPU_followon=False,
        previous_v2_failure_324missing_preserved=True,current_fleet_busy_qualified=False,
        next="NewGPUv3 sourceclaim must apply pinned patch and bind these actual CPU original receipts; no replay of v2 or this CPU scope.")


if __name__ == "__main__":
    print(json.dumps(audit(),allow_nan=False))
