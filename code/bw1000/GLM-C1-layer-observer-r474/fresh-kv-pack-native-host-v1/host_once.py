"""Run or reconcile the original GLM fresh-KV Native CPU step."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import runpy
import shlex
import struct
import subprocess
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
LABEL = "GLM474-fresh-kv-pack-Native-CPU-v1"
REQUEST = "e5ead062820a840906c74fce7294023b"
KERNEL_REQUEST = "b9a7f0b78c566911ed40004fc906ce44"
KERNEL_LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v1"
PARENT, NODE, ZONE = "1103606", "f12r2n13", "O01"
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=16384, DCU=0)
PLANNED = dict(fresh_M=2, HOST_compile=2, GPU_compile=0, reused_objects=30, link_objects=33, links=1)
WORKER_PUBLICATION = "34da8adfefc5f08d3ceef01b462d546e7d2b9c69"
SOURCE_CLAIM = Path("coordination/tasks/glm-r474-fresh-kv-pack-native-cpu-v1.json")
CLAIM_PIN = "2691b803a2b8d2b54f5f52e7013f58aa2c15749ed0f02a70dc1baf400595089f"
WORKER = Path("tools/run_glm474_fresh_kv_native_once.py")
RECIPE = Path("tools/prepare_glm474_fresh_kv_native_build.py")
MEMBERSHIP = Path("tools/read_training_slurm_membership.py")
LEDGER = Path("coordination/training/training80-active-node-task-map-v1.json")
ACCOUNTING = Path("code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py")
KERNEL_HOST = Path("code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-host-compile-v1/host_once.py")
KERNEL_HOST_CLAIM = "005a011e94cbe1fa5f6f11a735d1deb1afd9ee88656632656b48e091c576a5f9"
ENV = Path("code/bw1000/GLM-C1-layer-observer-r474/depth-stratified-observer-CPU-v1/ENV.sh")
PINS = {
    WORKER: "2328ebe0943eecc5b0de6402c33d48e28c4e933c75d368be109c312f8bda1e2b",
    RECIPE: "12d864be8c3cbd719845aa94be41463116ca521835b445e007dcfbe57b6c1bcd",
    KERNEL_HOST: "490c8d242b28e9a1cbe7eb8e4c15c4bdcf0b9534c953c77f43cfb877d6341db4",
    MEMBERSHIP: "d2973a1841a0e8cc2fcf616fee383e8ea95b783d4914f203889a6dcafedc22b3",
    LEDGER: "36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9",
    ACCOUNTING: "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    ENV: "147ea951bfe67a82ba39ef62e2e695e4ae7d21f8cc03f95b60054527ed960db9",
}
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
CHECKOUT = f"/refactor/incoming/{LABEL}-checkout"
PROOF = f"/refactor/results/{LABEL}-controller/kernel-reconciliation.json"
AFFINITY_KIND = "GLM474_NATIVE_ASSIGNED_CPU"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_bytes(raw):
    return sha256(raw).hexdigest()


def read(path, expected=None, limit=2**21):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    require(0 < len(raw) <= limit, f"missing or oversized original evidence: {path.name}")
    require(expected is None or hash_bytes(raw) == expected, f"original SHA mismatch: {path.name}")
    return raw


def save(path, row):
    raw = (json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hash_bytes(raw)


def frozen():
    claim = json.loads(read(REPO / SOURCE_CLAIM, CLAIM_PIN, 16384))
    require(claim["request"] == REQUEST and claim["label"] == LABEL and claim["placement"] == PLACEMENT
            and claim["planned"] == PLANNED and claim["kernel_prerequisite_request"] == KERNEL_REQUEST
            and (claim["whole_seconds"], claim["work_seconds"]) == (1800, 1680)
            and claim["CPU_GO"] is True and claim["GPU_GO"] is False and claim["Model_GO"] is False
            and claim["replay_allowed"] is False and claim["runner_SHA256"] == PINS[WORKER]
            and claim["recipe_tool_SHA256"] == PINS[RECIPE], "published Native scope differs")
    raw = {path: read(REPO / path, pin) for path, pin in PINS.items()}
    waiter, member = {}, {}
    for path, dst in ((ACCOUNTING, waiter), (MEMBERSHIP, member)):
        dst.update(__name__="frozen_shared_slurm")
        exec(compile(raw[path], str(REPO / path), "exec"), dst)
    kernel = runpy.run_path(str(REPO / KERNEL_HOST))
    ledger = json.loads(raw[LEDGER])
    groups = {parent: {zone: group["allocations"][parent] for zone, group in ledger["groups"].items()}
              for parent in ("1087656", PARENT)}
    return kernel, waiter, member, groups


def accounting_row(raw):
    require(len(raw) <= 2**21, "accounting output exceeded its bound")
    rows = [line.split("|") for line in raw.decode().splitlines() if line]
    require(all(len(row) == len(COLUMNS) for row in rows), "ambiguous accounting columns")
    matches = [row for row in rows if row[1] == LABEL]
    require(len(matches) <= 1, "ambiguous original Native step")
    row = matches[0] if matches else None
    require(row is None or (re.fullmatch(r"1103606\.[0-9]+", row[0]) and row[2].strip() and row[5] == NODE),
            "Native step appeared under different parent or node")
    return row


def query(timeout):
    return accounting_row(subprocess.check_output(
        ["sacct", "-n", "-P", "-j", "1087656,1103606", "-S", "2026-11-20T00:00:00", "-o", ",".join(COLUMNS)],
        timeout=timeout))


def slot(snapshot, cleared_step, after_epoch):
    parents = snapshot["parents"]
    require(set(parents) == {"1087656", PARENT} and snapshot["membership_qualified"] is True
            and all(row["owned_by_observer"] is True for row in parents.values())
            and snapshot["observed_epoch"] >= after_epoch, "unqualified independent two-parent ownership observation")
    row = parents[PARENT]["coverage_by_group_and_node"][ZONE][NODE]
    require(all(cleared_step not in parent["active_steps"] for parent in parents.values())
            and row["state"] == "NO_NUMBERED_STEP" and row["numbered_step_assigned_DCU_bounds"] == [0, 0],
            "original O01 step or conflicting work remains active")
    return {"node": NODE, "parent": PARENT, "cleared_original_step": cleared_step,
            "observed_epoch": snapshot["observed_epoch"], "original_step_absent_from_both_parents": True,
            "node_without_numbered_step": True, "scope": "ZERO_GRES_SLURM_MEMBERSHIP_RELEASE"}


def kernel_gate(reconcile, observe, wall):
    proof = reconcile(KERNEL_HOST_CLAIM, reconcile=True)
    row = proof.get("row")
    require(proof.get("request") == KERNEL_REQUEST and proof.get("state") == "COMPILE_VERIFIED"
            and proof.get("independent_observation") is True and proof.get("actual_GPU_steps") == 0
            and row and len(row) == len(COLUMNS) and row[1:4] == [KERNEL_LABEL, "COMPLETED", "0:0"]
            and row[5] == NODE and re.fullmatch(r"1103606\.[0-9]+", row[0])
            and all(re.fullmatch("[0-9a-f]{64}", proof.get(key, "")) for key in
                    ("object_SHA256", "receipt_SHA256", "terminal_row_SHA256", "fresh_M_SHA256", "actual_MD_SHA256")),
            "original gfx936 object has no independently reconciled terminal")
    require(proof["terminal_row_SHA256"] == hash_bytes(json.dumps(row, separators=(",", ":")).encode()),
            "unbound original kernel accounting SHA")
    after = wall()
    return proof, slot(observe(), row[0], after)


def parent_expiry(fields):
    lines = subprocess.check_output(["scontrol", "show", "job", PARENT, "--oneliner"], timeout=8).decode().splitlines()
    require(len(lines) == 1, "unavailable Native parent allocation")
    row = fields(lines[0])
    owner = re.fullmatch(r"[^()]+\(([0-9]+)\)", row.get("UserId", ""))
    require(row.get("JobId") == PARENT and row.get("JobState") == "RUNNING" and owner
            and int(owner[1]) == os.getuid(), "Native parent owner changed")
    end = datetime.fromisoformat(row["EndTime"]).timestamp()
    require(math.isfinite(end), "invalid Native parent end")
    return end


def command(root, checkout, run_path, epoch, kernel_sha, seconds):
    bind = [
        f"{root}:/refactor:ro", f"{root / 'candidates'}:/refactor/candidates:rw",
        f"{root / 'candidates' / KERNEL_LABEL}:/refactor/candidates/{KERNEL_LABEL}:ro",
        f"{checkout}:{CHECKOUT}:ro", f"{run_path / 'kernel-reconciliation.json'}:{PROOF}:ro",
        "/opt/hyhal:/opt/hyhal:ro",
    ]
    require(all("," not in str(path) and ":" not in str(path) for path in (root, checkout, run_path)),
            "bind source path is not canonical POSIX")
    container = ["/public/software/apps/apptainer/1.3.4/bin/apptainer", "exec", "--rocm", "--no-home", "--pwd", "/"]
    for mount in bind:
        container.extend(["--bind", mount])
    bootstrap = "\n".join((
        "from pathlib import Path",
        "import json,os,runpy,sys",
        "p=Path(sys.argv[1]);sys.argv=sys.argv[1:]",
        "mask=sorted(os.sched_getaffinity(0))",
        "step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')",
        f"print(json.dumps(dict(kind={AFFINITY_KIND!r},step=step,cpus=mask,"
        "cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK')),sort_keys=True),flush=True)",
        "if len(mask)!=4:",
        "    raise RuntimeError('Native worker requires exactly four Slurm-bound CPUs')",
        "sys.path.insert(0,str(p.parent));runpy.run_path(str(p),run_name='__main__')",
    ))
    python = ["/usr/bin/python3", "-I", "-S", "-B", "-u", "-c", bootstrap, f"{CHECKOUT}/{WORKER.as_posix()}",
              "--claim-sha256", CLAIM_PIN, "--kernel-reconciliation", PROOF,
              "--kernel-reconciliation-sha256", kernel_sha, "--epoch", str(epoch),
              "--deadline", str(epoch + 1800), "--work-deadline", str(epoch + 1680)]
    shell = f"source {shlex.quote(f'{CHECKOUT}/{ENV.as_posix()}')}; exec " + shlex.join(python)
    duration = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:00"
    return ["srun", "--jobid=" + PARENT, "--job-name=" + LABEL, "--exact", "--immediate=5",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--cpu-bind=threads", "--mem=16384M", "--gres=none",
            "--nodelist=" + NODE, "--time=" + duration, *container,
            str(root / "containers/dtk26-sandbox-r1"), "/bin/bash", "--noprofile", "--norc", "-c", shell]


def assigned_cpu_evidence(log_path, step):
    matches = []
    for raw in read(log_path).splitlines():
        if AFFINITY_KIND.encode() in raw:
            record = json.loads(raw)
            require(record.get("kind") == AFFINITY_KIND, "unexpected CPU binding record")
            matches.append((raw, record))
    require(len(matches) == 1, "expected one assigned CPU binding observation")
    raw, record = matches[0]
    cpus = record.get("cpus")
    require(record.get("step") == step and record.get("cpus_per_task") in (None, "4")
            and isinstance(cpus, list) and len(cpus) == 4 and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
            and cpus == sorted(set(cpus)), "original Native step lacks exact four-CPU affinity")
    return dict(assigned_CPU_ids=cpus, affinity_record_SHA256=hash_bytes(raw))


def verify_terminal(output, row, epoch, kernel, log_path):
    if row[2:4] != ["COMPLETED", "0:0"]:
        return {}
    require(len(row) == len(COLUMNS) and row[1] == LABEL and row[5] == NODE
            and re.fullmatch(r"1103606\.[0-9]+", row[0]), "foreign Native terminal")
    pairs = [text.split("=", 1) for text in row[6].split(",")]
    require(all(len(part) == 2 for part in pairs), "invalid original Native TRES")
    tres = dict(pairs)
    require(len(tres) == len(pairs) and tres.get("cpu") == "4" and tres.get("node") == "1"
            and tres.get("mem") in {"16G", "16384M", "16777216K"} and row[4].isdecimal() and int(row[4]) <= 1800
            and all(v.isdecimal() and int(v) == 0 for k, v in tres.items() if k.startswith("gres/")),
            "actual Native terminal accounting differs")
    affinity = assigned_cpu_evidence(log_path, row[0])
    raw = read(output / "CPU-result.json")
    worker = json.loads(raw)
    expected = dict(label=LABEL, request=REQUEST, step=row[0], node=NODE, epoch=epoch, deadline=epoch + 1800,
                    work_deadline=epoch + 1680, claim_SHA256=CLAIM_PIN, planned=PLANNED, actual=PLANNED,
                    GPU_GO=False, Model_GO=False, passed=True, completed=True, phase="complete", replay_allowed=False)
    require(all(type(worker.get(key)) is type(value) and worker[key] == value for key, value in expected.items()),
            "actual Native worker disagrees with original scope")
    recipe_raw = read(output / "BUILD_RECIPE.json", worker["recipe_SHA256"])
    recipe = json.loads(recipe_raw)
    expected_commands = [(f"{phase}-{unit}", recipe["host"][unit][key])
                         for phase, key in (("fresh", "preprocess"), ("compile", "compile")) for unit in ("u15", "u17")]
    expected_commands.append(("link", recipe["link"]))
    require(recipe["planned"] == PLANNED and recipe["original_step"] == "1103606.308"
            and recipe["kernel"] == dict(path=f"/refactor/candidates/{KERNEL_LABEL}/fresh_kv_pack.o",
                                         SHA256=kernel["object_SHA256"], slurm_step=kernel["row"][0])
            and set(recipe["host"]) == {"u15", "u17"} and len(worker["commands"]) == len(expected_commands)
            and all(cmd["phase"] == phase and cmd["argv"] == argv and cmd["returncode"] == 0
                    and cmd["owned_child_reaped"] is True
                    for cmd, (phase, argv) in zip(worker["commands"], expected_commands)),
            "original compiler phases or child completion differ")
    first, after = (worker[name + "_dependency_evidence"] for name in ("fresh", "compile"))
    linked = worker["linked"]
    require(first["phase"] == "fresh" and after["phase"] == linked["phase"] == "actual"
            and all(row["recipe_SHA256"] == worker["recipe_SHA256"] for row in (first, after, linked))
            and first["hosts"] == after["hosts"] == linked["hosts"] and set(linked["hosts"]) == {"u15", "u17"}
            and linked["strong_exports"] == 283 and linked["providers"] == 71 and linked["explicit_Runtime_calls"] == 0
            and linked["implicit_ldd_initialization"] == "UNKNOWN", "original Native provider or dependency proof differs")
    dependency_hashes = {}
    for unit in ("u15", "u17"):
        expected_target = f"/refactor/candidates/{LABEL}/{unit}.o"
        hashes, sets = {}, []
        for phase in ("before", "actual"):
            blob = read(output / f"{unit}.{phase}.d")
            target, separator, rest = blob.decode().replace("\\\n", " ").partition(":")
            deps = set(shlex.split(rest))
            require(target.strip() == expected_target and separator and deps and all(p.startswith("/") for p in deps),
                    "original Native M/MD target differs")
            hashes[phase] = hash_bytes(blob)
            sets.append(deps)
        require(sets[0] == sets[1] and len(sets[0]) == linked["hosts"][unit]["paths"],
                "original Native M/MD paths differ")
        dependency_hashes[unit] = hashes
    binary = read(output / "libGLM_fresh_kv_pack.so", linked["library_SHA256"], 16*1024**2)
    require(len(binary) >= 64 and binary[:7] == b"\x7fELF\x02\x01\x01"
            and struct.unpack_from("<HHI", binary, 16) == (3, 62, 1)
            and len(binary) == linked["library_bytes"], "original Native ELF differs")
    return dict(worker_SHA256=hash_bytes(raw), recipe_SHA256=hash_bytes(recipe_raw),
                library_SHA256=hash_bytes(binary), library_bytes=len(binary),
                terminal_row_SHA256=hash_bytes(json.dumps(row, separators=(",", ":")).encode()),
                dependency_file_SHA256=dependency_hashes, **affinity,
                actual=PLANNED)


def inspect_original(run_path, output, accounting, is_terminal):
    raw = read(run_path / "intent.json")
    intent = json.loads(raw)
    epoch = intent["epoch"]
    require(intent["request"] == REQUEST and intent["claim_SHA256"] == CLAIM_PIN
            and intent["worker_publication"] == WORKER_PUBLICATION and intent["placement"] == PLACEMENT
            and intent["replay_allowed"] is False and type(epoch) is float and math.isfinite(epoch) and epoch > 0
            and (intent["deadline"], intent["work_deadline"]) == (epoch+1800, epoch+1680),
            "not the original Native intent")
    kernel = json.loads(read(run_path / "kernel-reconciliation.json", intent["kernel_reconciliation_SHA256"], 16384))
    read(run_path / "kernel-cleanup.json", intent["kernel_cleanup_SHA256"], 16384)
    row = accounting(8)
    prior_path = run_path / "accounting-result.json"
    prior = json.loads(read(prior_path)) if prior_path.exists() else {}
    if prior:
        require(prior["request"] == REQUEST and prior["accounting_columns"] == COLUMNS
                and prior["actual_GPU_steps"] == 0, "prior Native controller scope differs")
    if prior.get("terminal_visible") is True:
        require(prior["row"] == row, "prior Native accounting diverged")
    result = dict(request=REQUEST, state="UNKNOWN", row=row, terminal_visible=False, actual_GPU_steps=0,
                  accounting_columns=COLUMNS, intent_SHA256=hash_bytes(raw), independent_observation=True)
    if is_terminal(row):
        result.update(state="TERMINAL_UNVERIFIED", terminal_visible=True)
        verified = verify_terminal(output, row, epoch, kernel, run_path / "srun.log")
        if verified:
            if prior.get("state") == "NATIVE_VERIFIED":
                require(all(prior.get(key) == value for key, value in verified.items()), "original Native SHA changed")
            result.update(verified, state="NATIVE_VERIFIED")
    return result


def launch(run_path, output, checkout, *, reconcile_kernel, observe, expires, accounting, runner,
           wait_terminal, wall=time.time, monotonic=time.monotonic):
    require(not any(p.exists() or p.is_symlink() for p in (run_path, output)),
            "original Native intent or worker namespace exists; reconcile")
    proof, cleanup = kernel_gate(reconcile_kernel, observe, wall)
    epoch, started = wall(), monotonic()
    deadline, work_deadline = epoch + 1800, epoch + 1680

    def left(reserve=0):
        return min(deadline-wall(), 1800-(monotonic()-started))-reserve

    placement = slot(observe(), proof["row"][0], epoch)
    end = expires()
    require(end >= deadline, "parent allocation expires before this Native epoch")
    check_budget = min(8, left(35))
    require(check_budget > 0, "original Native preflight budget exhausted")
    require(accounting(check_budget) is None, "original Native step already present; reconcile")
    seconds = 60 * (min(1740, int(left(40))) // 60)
    require(seconds >= 60 and work_deadline-wall() > 25 and seconds - (work_deadline-wall()) > 15,
            "insufficient work/Slurm reserve in original epoch")

    run_path.mkdir(mode=0o700)
    kernel_pin = save(run_path / "kernel-reconciliation.json", proof)
    cleanup_pin = save(run_path / "kernel-cleanup.json", cleanup)
    args = command(ROOT, checkout, run_path, epoch, kernel_pin, seconds)
    intent = dict(request=REQUEST, claim_SHA256=CLAIM_PIN, worker_publication=WORKER_PUBLICATION,
                  placement=PLACEMENT, epoch=epoch, deadline=deadline, work_deadline=work_deadline,
                  kernel_step=proof["row"][0], kernel_reconciliation_SHA256=kernel_pin,
                  kernel_cleanup_SHA256=cleanup_pin, fresh_slot=placement, parent_end=end,
                  slurm_seconds=seconds, argv=args, replay_allowed=False)
    intent_pin = save(run_path / "intent.json", intent)
    result = dict(request=REQUEST, state="UNKNOWN", actual_GPU_steps=0, accounting_columns=COLUMNS,
                  terminal_visible=False, intent_SHA256=intent_pin, row=None)

    def read_accounting():
        remaining = min(8, left(3))
        if remaining <= 0:
            raise TimeoutError("original Native accounting deadline elapsed")
        return accounting(remaining)

    try:
        require(left(30) >= seconds and work_deadline-wall() > 25, "original epoch expired before launch")
        with (run_path / "srun.log").open("xb") as log:
            save(run_path / "exec-attempt.json", dict(request=REQUEST, intent_SHA256=intent_pin,
                                                     attempted_epoch=wall(), replay_allowed=False))
            try:
                result["client_rc"] = runner(args, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, timeout=left(25), check=False).returncode
            except subprocess.TimeoutExpired:
                result["client_timeout"] = True
        terminal = wait_terminal(read_accounting, monotonic()+min(18, left(2)), max_observations=6)
        result.update(terminal)
        if terminal["terminal_visible"]:
            result["state"] = "TERMINAL_UNVERIFIED"
            verified = verify_terminal(output, terminal["row"], epoch, proof, run_path / "srun.log")
            if verified:
                result.update(verified, state="NATIVE_VERIFIED")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result["error"] = f"{type(error).__name__}: {str(error)[:230]}"
    finally:
        save(run_path / "accounting-result.json", result)
    return result


def execute(*, reconcile=False):
    kernel, waiter, member, groups = frozen()
    require(REPO.resolve(strict=True) == REPO and REPO.is_relative_to(ROOT),
            "expected canonical published checkout within the existing project")
    run_path = ROOT / "results" / (LABEL + "-controller")
    output = ROOT / "candidates" / LABEL
    if reconcile:
        return inspect_original(run_path, output, query, waiter["is_terminal"])
    return launch(run_path, output, REPO, reconcile_kernel=kernel["execute"],
                  observe=lambda: member["snapshot"](groups, owner_uid=os.getuid()),
                  expires=lambda: parent_expiry(member["fields"]), accounting=query, runner=subprocess.run,
                  wait_terminal=waiter["wait_terminal"])


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile", action="store_true", help="read the original intent and observe accounting once")
    args = parser.parse_args(args)
    row = execute(reconcile=args.reconcile)
    print(json.dumps(row, allow_nan=False), flush=True)
    return int(row["state"] != "NATIVE_VERIFIED")


if __name__ == "__main__":
    raise SystemExit(main())
