"""Resume only the verified BF16 standby on f12r2n11 after original .3087 returns."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
import time
from types import FunctionType


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
STEM = "training80-o01-3087-guarded-bf16-v1"
CLAIM = REPO / "coordination/tasks" / f"{STEM}.json"
BASE = REPO / "tools/run_training80_prebridge_five_guarded_refill_once.py"
BASE_PIN = "7b88d30a4af7eff4f25fbcc663de674af328a03de74e364c517752bd599d306c"
BASE_CLAIM_PIN = "c4c18f644e6ebc2a99ea0e19d2261a9cbbe47efe743e79dd5ae8f79fea4ce974"
READER = LOCAL / "read-training80-o01-original3087-dual-source-once.py"
READER_PIN = "e86a759960f20f27d072976dd48ce869d772a6517153ad9bc9443f2c1037b1e8"
RUN = REPO / "results/bw1000/2026-11-25/training80-o01-original3001-to3087-verified-standby-refill.json"
GAP = LOCAL / "training80-1224-exact-gap-read-c7b27064889f4aaea7d9f82bc0931980.json"
RUN_PIN = "5c0beceb0ea1da36cb694d0292d266ac8b7efe18479f180635c02798aa1e9fc6"
GAP_PIN = "9d30667edafba6f177154f3cfc1f3ef751209072540d592a8ebdbad465a481e7"
NODE, PARENT = "f12r2n11", "1103606"
PREVIOUS, CLEAR, PROTECTED = "1103606.3087", "1103606.3086", "1103606.2424"
OLD_REQUEST = "6c90973adba441c593855dc5393121d4"
OLD_CHECK = {
    "read_id": "4c49ef0f524844d4882279eb8f8f1cc9",
    "receipt_SHA256": "05729c4f67c4be76d565593237355ad1d74c2355214733ede3058f6f8408d5b6",
    "terminal_SHA256": "6641d06e879dbc56b6b2948b8a802f3aecaa62c3a579ac4df45c7a3ac0df41e3",
}
ORDINARY = REPO / "tools/run_training80_ordinary58_1140_refill_once.py"
ORDINARY_PIN = "9cdf04cf946058c837ff9b5101f9eff56e969c595ae3b2397fe064834f2c4ba2"
PROTECTIONS = {
    ".training80-o01-successor-grant-ea2b693489164cb199a95fb844a9e243/grant.json":
        "452d40f124d6c8cbaf152ad826c940e2c30108e654f89bd5b539c5813a577a03",
    ".training80-o01-pending-274608fc6e0d410a8f545287843cb887/released.json": None,
    ".training80-o01-pending-e23d24ea0c9c4d00af06154d82dc22e8/released.json":
        "b00fc3b0b523abd1dd9ac7106d7ad62ffb84867f99647ce019a0cad8c61ce5f9",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def utc_row(row):
    cst = timezone(timedelta(hours=8))
    converted = [datetime.fromisoformat(value).replace(tzinfo=cst).astimezone(timezone.utc).replace(tzinfo=None).isoformat()
                 for value in row[7:9]]
    return row[:7] + converted


def load():
    require(BASE.resolve(strict=True) == BASE and digest(BASE.read_bytes()) == BASE_PIN, "guarded T05/T07 base changed")
    base = runpy.run_path(str(BASE))
    config = base["prepared"](json.loads(base["pin"](base["CLAIM"], BASE_CLAIM_PIN)))
    base["pin"](ORDINARY, ORDINARY_PIN)
    active = runpy.run_path(str(ORDINARY))["plan"]()
    require(not any(NODE == item["node"] for group in active[-1] for item in group), "specialist overlaps ordinary roster")
    raw = base["pin"](CLAIM)
    claim = json.loads(raw)
    require((claim["scope"], claim["state"], claim["node"], claim["parent"], claim["previous_request"], claim["previous_gpu_step"],
             claim["previous_zero_GRES"], claim["protected_historical_zero_GRES"], claim["max_invocations"], claim["replay_allowed"])
            == (STEM, "ORIGINAL_RETURN_VERIFIED_FRESH_ADMISSION_PENDING", NODE, PARENT, OLD_REQUEST, PREVIOUS, CLEAR, PROTECTED, 1, False)
            and claim["previous_run_SHA256"] == RUN_PIN and claim["confirmed_gap_read_SHA256"] == GAP_PIN
            and claim["original_natural_return"] == OLD_CHECK
            and claim["onboard"]["GPU"] == 8 and re.fullmatch(r"[a-f0-9]{32}", claim["request"])
            and claim["request"] != OLD_REQUEST, "fixed singleton O01 scope differs")
    public = json.loads(base["pin"](RUN, RUN_PIN))
    gap = json.loads(base["pin"](GAP, GAP_PIN))
    snap = gap["result"]
    latest, = snap["accounting"][PREVIOUS]
    expected_gpu = [PREVIOUS, f"training80-dcu-standby-{OLD_REQUEST[:12]}-{NODE}", "COMPLETED", "0:0", NODE, base["GPU_TRES"]]
    placement = snap["targets"][NODE]
    require(gap["rc"] == 0 and latest[:6] == expected_gpu and latest[-1] == "2026-11-25T02:15:00"
            and placement["observed_owner"] is True and placement["parent"] == PARENT
            and placement["controller"] == placement["queue"] == [], "original .3087 actual terminal or dual view differs")
    old_gpu = latest[:4] + ["UNMEASURED"] + latest[4:]
    old_cpu, old_protected = (utc_row(next(row for row in public["actual_accounting"] if row[0] == key)) for key in (CLEAR, PROTECTED))
    require(public["request"] == OLD_REQUEST and public["new_actual_GPU_step"] == PREVIOUS and public["new_pre_zero_GRES"] == CLEAR
            and public["parent"] == PARENT and public["node"] == NODE and public["budget_seconds"] == 10568
            and public["prior_release_unchanged"] is True and public["original_guard_SHA256"] == next(iter(PROTECTIONS.values())),
            "old qualifying O01 run differs")
    archive = public["evidence"][0]
    progress = json.loads(base["pin"](LOCAL / archive["file"], archive["SHA256"]))["observed"]
    first, second = progress["first"], progress["second"]
    pci = sorted(progress["advancing"])
    require((progress["request"], progress["step"], progress["node"], progress["pre"], progress["worker_seconds"])
            == (OLD_REQUEST, PREVIOUS, NODE, CLEAR, 10568) and progress["direct_no_new_bridge"] is True
            and progress["gpu_running_same"] is True and progress["fresh_pre_actual_C0"] is True
            and progress["old_admission_preserved"] is True
            and len(pci) == len(set(pci)) == 8 and set(pci) == set(first["rows"]) == set(second["rows"])
            and first["worker_start"] == second["worker_start"] and first["worker_pid"] == second["worker_pid"] > 0,
            "old independent worker identity differs")
    pins = {k: progress[f"{k}_SHA256"] for k in ("worker", "expected")}
    base["pin"](READER, READER_PIN)
    reader = runpy.run_path(str(READER))
    root_rows = (old_gpu, old_cpu, old_protected)
    return base, config, claim, digest(raw), root_rows, pins, pci, first, reader


def terminal_receipt(material, read_id, proof_pin, *, fresh=False):
    base, _, _, _, (gpu_seed, cpu_seed, protected), baseline, pci, start, reader = material
    wrapper = json.loads(base["pin"](reader["receipt"](read_id), proof_pin))
    if fresh:
        now = time.monotonic()
        require(read_id != OLD_CHECK["read_id"] and abs(time.time()-now-wrapper["boot_epoch_anchor"]) < 2
                and 0 <= now-wrapper["received_monotonic"] <= 45, "a new independent predecessor read is required")
    result = wrapper["observed"]
    require(wrapper["rc"] == 0 and wrapper["read_id"] == read_id and wrapper["replay_allowed"] is False
            and wrapper["Slurm_mutations"] == 0 and 0 <= wrapper["local_roundtrip_s"] <= 29
            and wrapper["source_SHA256"] == reader["digest"](reader["program"]())
            and result["request"] == OLD_REQUEST and result["owner"] is result["alloc_320"] is True
            and result["parent_state"] == "RUNNING" and result["live"] == result["squeue"] == {}
            and result["prior_release_unchanged"] is True and result["grant_SHA256"] == next(iter(PROTECTIONS.values())),
            "old worker actual terminal, protections or two scheduler views unverified")
    gpu, = result["accounting"][PREVIOUS]
    cpu, = result["accounting"][CLEAR]
    fenced, = result["accounting"][PROTECTED]
    require(gpu[:4] == gpu_seed[:4] and gpu[5:] == gpu_seed[5:] and gpu[4].isdecimal() and int(gpu[4]) >= 10568
            and cpu == cpu_seed and fenced == protected, "exact .3087/.3086/.2424 accounting differs")
    expected, worker, terminal = (result[key]["document"] for key in ("expected", "worker", "terminal"))
    require({k: result[k]["SHA256"] for k in baseline} == baseline
            and expected["request"] == OLD_REQUEST and expected["label"] == gpu[1] and expected["parent"] == PARENT
            and expected["seconds"] == 10568 and expected["node"] == worker["node"] == terminal["node"] == NODE
            and worker["step"] == terminal["step"] == PREVIOUS and worker["synthetic"] is True
            and worker["worker_pid"] == start["worker_pid"] > 0 and worker["epoch"] == start["worker_start"]
            and sorted(expected["pci"]) == sorted(worker["pci"]) == pci
            and terminal["state"] == "WORKER_RETURNED" and worker["epoch"] + expected["seconds"] <= terminal["epoch"] < result["epoch"],
            "same old worker natural return is not proved")
    terminal_pin = result["terminal"]["SHA256"]
    require(terminal_pin == OLD_CHECK["terminal_SHA256"], "old terminal digest differs")
    return (gpu, cpu, fenced), dict(baseline, terminal=terminal_pin)


def bound(base, name):
    original = base[name]
    paths = FunctionType(base["paths"].__code__, dict(base["paths"].__globals__, STEM=STEM))
    return FunctionType(original.__code__, dict(original.__globals__, STEM=STEM, FIVE={NODE: (PREVIOUS, CLEAR)}, paths=paths))


def render(material, publication, prepared_rows=None, old_pins=None):
    base, config, claim, _, seed, baseline, pci, _, _ = material
    gpu, cpu, protected = prepared_rows or seed
    pins = old_pins or dict(baseline, terminal="0" * 64)
    config[3][NODE] = (gpu, cpu, pci)
    program = bound(base, "render")(NODE, {"targets": {NODE: {"request": claim["request"]}}}, config, publication)
    checks = dict(PROTECTIONS)
    checks.update({f"training80-dcu-standby-{OLD_REQUEST}/{NODE}/{name}.json": value for name, value in pins.items()})
    anchor = "            check_previous(account(PREVIOUS))"
    require(program["launch"].count(anchor) == 1, "guarded shared lock insertion moved")
    gate = (
        "            for rel, expected in " + repr(checks) + ".items():\n"
        "                old_file = RESULTS / rel\n"
        "                if expected is None:\n"
        "                    require(not os.path.lexists(old_file), 'original absent release changed')\n"
        "                    continue\n"
        "                require(old_file.resolve(strict=True) == old_file, 'old protected path differs')\n"
        "                with old_file.open('rb') as file:\n"
        "                    raw = file.read(8193)\n"
        "                require(0 < len(raw) <= 8192 and sha256(raw).hexdigest() == expected, 'prior worker or release changed')\n"
        "            protected = account('1103606.2424')\n"
        "            require(protected == " + repr(protected[:4]+protected[5:])+ ", 'protected O01 zero-GRES terminal differs')\n"
    )
    prefix = "import os\nos.environ.update(TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')\n"
    allowance = "({OBSERVER_STEP, own} if own else {OBSERVER_STEP})"
    require(program["launch"].count(allowance) == 1, "dual scheduler view guard changed")
    program["launch"] = prefix + program["launch"].replace(anchor, gate + anchor, 1).replace(allowance, "({own} if own else set())", 1)
    program["observe"] = prefix + program["observe"]
    for name in ("launch", "observe"):
        compile(program[name], "<O01 exact 3087 guarded successor>", "exec")
    return program


def publish(revision, base, config):
    base["verify_publication"](revision, config)
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]
    for path in (Path(__file__).resolve(), CLAIM, BASE):
        ref = revision + ":" + path.relative_to(REPO).as_posix()
        require(subprocess.check_output([*git, "cat-file", "blob", ref], cwd=REPO, timeout=10) == path.read_bytes(),
                "exact published successor differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--publication")
    modes.add_argument("--observe-original", action="store_true")
    parser.add_argument("--terminal-receipt", nargs=2, metavar=("READ_ID", "SHA256"))
    args = parser.parse_args()
    require(bool(args.publication) == bool(args.terminal_receipt), "actual original worker terminal read id and SHA required")
    material = load()
    base, config, claim, claim_pin, *_ = material
    rows, pins = terminal_receipt(material, OLD_CHECK["read_id"], OLD_CHECK["receipt_SHA256"])
    if args.check:
        program = render(material, None, rows, pins)
        result = dict(state="ORIGINAL_RETURN_VERIFIED_FRESH_ADMISSION_PENDING", one_request=claim["request"], claim_SHA256=claim_pin,
                      programmed_worker_seconds=program["worker_seconds"], compiled_only_SHA256=digest(program["launch"].encode()),
                      predecessor=PREVIOUS, GPU=0, Slurm=0, SCNet=0, Git=0)
    elif args.observe_original:
        result = bound(base, "observe")(NODE, render(material, None, rows, pins), config)
    else:
        publish(args.publication, base, config)
        ranks = base["live_authority"](config)
        rows, pins = terminal_receipt(material, *args.terminal_receipt, fresh=True)
        result = bound(base, "dispatch")(NODE, render(material, args.publication, rows, pins), args.publication,
                                          claim_pin, config, ranks)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
