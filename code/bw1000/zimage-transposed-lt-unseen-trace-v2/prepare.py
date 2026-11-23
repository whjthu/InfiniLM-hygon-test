"""Render the corrected .1985 tracer with a new exclusive execution envelope."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = HERE.with_name("zimage-transposed-lt-unseen-dispatch-v1")
REQUEST = "0f3f53e006af491da5e4e6d5ace1df21"
LABEL = "O01-zimage-transposed-lt-unseen-solution-v2"
PREVIOUS_REQUEST = "4241e3d58506499cb859abead6f937de"
PREVIOUS_LABEL = "O01-zimage-transposed-lt-unseen-solution-v1"
OLD_FILES = {
    "contract.py": "956c0892690a554fb31549051797f9dff82d851f1c4c79e7a3e7a2e07059f148",
    "device_entry.py": "349d46d348d89a6ea04077c64749754640bf3b6c4ccc0d4430b19f6b5466b090",
    "entry.py": "c92e9c9c9a88f4a3321a206d2423089bfd285fb28686b72e16e0b534a0434ffa",
    "node_inventory.py": "dfb047f40a1f21e9ff606afeb95e60026f9d350b5bb35e3b447d8e5b86fa7efc",
    "worker.py": "acae43f1548c072d302e58c3213bebd5d64a10fe565e88cb0b01e318a6652986",
    "DISPATCH.json": "4c080c90a18d52f025ee7555a80bc49478a7031bcf8944c13b186218047507ba",
}
EVIDENCE = REPO / "results/bw1000/2026-11-24"
OLD_TERMINAL = "O01-zimage-transposed-lt-unseen-original1985-profiler-harness-failure.json"
OLD_TERMINAL_SHA = "29d4e15aff494eb368266287f7d4a738d47547b64baa8afd4e3bf6f3598723e5"
CPU_PROBE = "O01-zimage-hipprof-original1991-zero-gres-start-and-scan-terminal.json"
CPU_PROBE_SHA = "033e9d0071aae3f647bcbd9702712a9398bc5436dd26fc06e2aadbda6f90ebfa"
STANDBY = "1103606.1989"
STANDBY_NAME = "training80-dcu-standby-fa66e1eb8cfc-f12r2n11"
HISTORICAL_EPOCH = 1795452858.5587769
HISTORICAL_SHA = "f3a9249672c18f2fc1928a3697435cf20cccaf00846a8402a247c07b52c54f46"
PCI = "0000:09:00.0"


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def pinned(path, expected):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or digest(raw) != expected:
        raise ValueError("exact source or receipt bytes changed: " + path.name)
    return raw


def pins(files):
    return {name: {"bytes": len(raw), "SHA256": digest(raw)} for name, raw in files.items()}


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("frozen source boundary changed: " + old[:80])
    return text.replace(old, new)


def evidence():
    failed = json.loads(pinned(EVIDENCE / OLD_TERMINAL, OLD_TERMINAL_SHA))
    cpu = json.loads(pinned(EVIDENCE / CPU_PROBE, CPU_PROBE_SHA))
    failed_rows = [row for row in failed["accounting"] if row[0] == "1103606.1985"]
    tests = cpu["runtime_summary"]
    if (failed["request"] != PREVIOUS_REQUEST or len(failed_rows) != 1
            or failed_rows[0][2:4] != ["FAILED", "1:0"] or failed["replay_allowed"] is not False
            or cpu["accounting"][:1] != ["1103606.1991"] or cpu["accounting"][2:4] != ["COMPLETED", "0:0"]
            or cpu["accounting"][5] != "f12r2n11" or cpu["DCU"] != 0
            or not all(tests[k]["pid_observed"] is True and tests[k]["returncode"] == 0 for k in ("help", "no_GPU_child"))
            or tests["actual_algorithm_or_GPU_qualification"] is not False):
        raise ValueError("original failure or actual zero-GRES start evidence differs")
    return {"original_1985_SHA256": OLD_TERMINAL_SHA, "start_boundary_1991_SHA256": CPU_PROBE_SHA}


def source_files():
    original = json.loads(pinned(ORIGINAL / "DISPATCH.json", OLD_FILES["DISPATCH.json"]))
    worker = pinned(ORIGINAL / "worker.py", OLD_FILES["worker.py"]).decode()
    worker = replace(worker, PREVIOUS_REQUEST, REQUEST)
    worker = replace(worker, PREVIOUS_LABEL, LABEL)
    worker = replace(worker, "import socket\n", "import socket\nimport stat\n")
    first, last = worker.index("def trace_size("), worker.index("def audit(")
    worker = worker[:first] + (HERE / "trace_harness.inc").read_text(encoding="utf8").rstrip() + "\n\n\n" + worker[last:]
    worker = replace(worker, '        report["trace_tool_rc"] = in_trace(command, args.out, scope["deadline_epoch"] - 12)',
        '        trace_harness = {}\n'
        '        report["trace_harness"] = trace_harness\n'
        '        report["trace_tool_rc"] = in_trace(command, args.out, scope["deadline_epoch"] - 12, trace_harness)')
    worker = replace(worker, "except (OSError, ValueError, KeyError) as error:",
                     "except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:")
    worker = replace(worker, '                  zero_profiled_throughput_pairs=True, training_performance=False,',
                     f'                  trace_harness_prerequisites={evidence()!r},\n'
                     '                  zero_profiled_throughput_pairs=True, training_performance=False,')
    files = {"worker.py": worker.encode(), "contract.py": pinned(ORIGINAL / "contract.py", OLD_FILES["contract.py"])}
    manifest = {"files": pins(files), "cpu_build": original["original_CPU_terminal"], "request": REQUEST, **evidence()}
    return {**files, "SOURCE_FILES.json": encode(manifest)}


def fresh_handoff(path, expected):
    raw = pinned(path, expected)
    value = json.loads(raw)
    first, second = value["first"], value["second"]
    active = value["last_slurm"][STANDBY_NAME]
    before = value["first_slurm"][STANDBY_NAME]
    pcis = set(second["rows"])
    observed = value["epoch"]
    times = (first["observed_epoch"], second["observed_epoch"], observed)
    if (expected == HISTORICAL_SHA or value["stable_8card_step"] is not True
            or active != before or (active["id"], active["nodes"], active["state"]) != (STANDBY, "f12r2n11", "RUNNING")
            or "gres/dcu=8" not in active["tres"].split(",") or first["step"] != second["step"] or second["step"] != STANDBY
            or not all(type(t) in (float, int) and math.isfinite(t) for t in times)
            or not HISTORICAL_EPOCH < times[0] < times[1] <= times[2]
            or len(pcis) != 8 or PCI not in pcis or set(first["rows"]) != pcis or set(value["advancing"]) != pcis
            or not all(second["rows"][pci]["calls"] > first["rows"][pci]["calls"] for pci in pcis)):
        raise ValueError("fresh exact original standby observation is required")
    return {"step": STANDBY, "name": STANDBY_NAME, "observed_epoch": observed, "SHA256": expected}


def execution_files(fresh):
    files = source_files()
    claim = json.loads(pinned(ORIGINAL / "DISPATCH.json", OLD_FILES["DISPATCH.json"]))
    host = pinned(ORIGINAL / "entry.py", OLD_FILES["entry.py"]).decode()
    host = replace(host, PREVIOUS_LABEL, LABEL)
    host = replace(host, PREVIOUS_REQUEST, REQUEST)
    host = replace(host, "1103606.1889", STANDBY)
    host = replace(host, "training80-dcu-standby-7d5ce2805aee-f12r2n11", STANDBY_NAME)
    host = replace(host, claim["files"]["SOURCE_FILES.json"]["SHA256"], digest(files["SOURCE_FILES.json"]))
    host = replace(host, "def accounting(name):", f"""def verify_fresh_handoff(claim):
    expected = {fresh!r}
    require(claim["fresh_handoff"] == expected and claim["handoff_observed_epoch"] == expected["observed_epoch"]
            and expected["observed_epoch"] <= time.time() < expected["observed_epoch"] + 300,
            "Fresh exact standby authority expired or differs")


def accounting(name):""")
    start_fresh = ('    require(claim["handoff_job"] == STANDBY_NAME\n'
                   '            and claim["handoff_observed_epoch"] <= time.time() < claim["handoff_observed_epoch"] + 300,\n'
                   '            "Fresh standby identity observation has expired")')
    host = replace(host, start_fresh, '    require(claim["handoff_job"] == STANDBY_NAME, "Wrong standby label")\n'
                   '    verify_fresh_handoff(claim)')
    host = replace(host, '        pinned_standby()\n        save("handoff-intent.json", {"request": REQUEST, "step": STANDBY, "node": NODE})',
        '        claim = json.loads((RUN / "DISPATCH.json").read_bytes())\n'
        '        pinned_standby()\n'
        '        verify_fresh_handoff(claim)\n'
        '        save("handoff-intent.json", {"request": REQUEST, "step": STANDBY, "node": NODE,\n'
        '                                    "authority": claim["fresh_handoff"]})')
    device = pinned(ORIGINAL / "device_entry.py", OLD_FILES["device_entry.py"]).decode()
    device = replace(device, OLD_FILES["worker.py"], digest(files["worker.py"]))
    files.update({"entry.py": host.encode(), "device_entry.py": device.encode(),
                  "node_inventory.py": pinned(ORIGINAL / "node_inventory.py", OLD_FILES["node_inventory.py"])})
    claim.update(label=LABEL, request=REQUEST, files=pins(files), handoff_step=STANDBY, handoff_job=STANDBY_NAME,
                 handoff_observed_epoch=fresh["observed_epoch"], fresh_handoff=fresh, **evidence())
    return {**files, "DISPATCH.json": encode(claim)}


def stage(output, files):
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="zimage-lt-trace-v2-", dir=output.parent) as tmp:
        folder = Path(tmp) / "bundle"
        folder.mkdir(mode=0o700)
        for name, raw in files.items():
            (folder / name).write_bytes(raw)
        folder.rename(output)
    return {"request": REQUEST, "label": LABEL, "files": pins(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    commands.add_parser("source").add_argument("output", type=Path)
    complete = commands.add_parser("envelope")
    complete.add_argument("fresh_evidence", type=Path)
    complete.add_argument("evidence_sha256")
    complete.add_argument("output", type=Path)
    args = parser.parse_args()
    files = source_files() if args.kind == "source" else execution_files(fresh_handoff(args.fresh_evidence, args.evidence_sha256))
    print(json.dumps(stage(args.output, files), sort_keys=True))
