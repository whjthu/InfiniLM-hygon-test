"""Run the Native controller through mocked accounting and worker receipts."""
import json
from pathlib import Path, PurePosixPath
import shlex
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_once as H


KERNEL_ROW = ["1103606.648", H.KERNEL_LABEL, "COMPLETED", "0:0", "8", H.NODE, "cpu=4,mem=8G,node=1"]
NATIVE_ROW = ["1103606.649", H.LABEL, "COMPLETED", "0:0", "77", H.NODE, "cpu=4,mem=16G,node=1"]


def proof():
    return dict(row=KERNEL_ROW, request=H.KERNEL_REQUEST, state="COMPILE_VERIFIED", independent_observation=True,
                actual_GPU_steps=0, terminal_row_SHA256=H.hash_bytes(json.dumps(KERNEL_ROW,
                  separators=(",", ":")).encode()), object_SHA256="a"*64, receipt_SHA256="b"*64,
                fresh_M_SHA256="c"*64, actual_MD_SHA256="c"*64)


def observed(*, active=None, foreign=False):
    one = dict(owned_by_observer=not foreign, active_steps={})
    two = dict(owned_by_observer=True, active_steps={active: {}} if active else {},
               coverage_by_group_and_node={"O01": {H.NODE: dict(
                   state="CPU_STEP_ONLY" if active else "NO_NUMBERED_STEP",
                   numbered_step_assigned_DCU_bounds=[0, 0])}})
    return dict(membership_qualified=True, observed_epoch=time.time(), parents={"1087656": one, "1103606": two})


def worker(output, intent, *, drift=False, file_drift=False):
    output.mkdir()
    epochs = {key: intent[key] for key in ("epoch", "deadline", "work_deadline")}
    # Explicit synthetic worker metadata. Host logic still opens, hashes and checks the evidence.
    elf = bytearray(64)
    elf[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", elf, 16, 3, 62, 1)
    binary = bytes(elf)
    (output / "libGLM_fresh_kv_pack.so").write_bytes(binary)
    hosts = {name: {"preprocess": ["g++", "-M", name], "compile": ["g++", "-c", name]}
             for name in ("u15", "u17")}
    recipe = dict(host=hosts, planned=H.PLANNED, link=["g++", "-shared", "fixture"], original_step="1103606.308",
                  kernel=dict(path=f"/refactor/candidates/{H.KERNEL_LABEL}/fresh_kv_pack.o",
                              SHA256=proof()["object_SHA256"], slurm_step=KERNEL_ROW[0]))
    for name in hosts:
        target = f"/refactor/candidates/{H.LABEL}/{name}.o"
        original = target + ": /header/1 /header/2 /header/3 /header/4 /header/5\n"
        (output / (name + ".before.d")).write_text(original)
        (output / (name + ".actual.d")).write_text(original.replace("/header/5", "/header/6")
                                                    if file_drift else original)
    pin = H.save(output / "BUILD_RECIPE.json", recipe)
    deps = {name: dict(paths=5, SHA256=("b" if drift else "a")*64) for name in hosts}
    fresh = dict(phase="fresh", recipe_SHA256=pin, hosts=deps)
    actual = dict(phase="actual", recipe_SHA256=pin, hosts={name: dict(paths=5, SHA256="a"*64) for name in hosts})
    link = dict(actual, library_SHA256=H.hash_bytes(binary), library_bytes=len(binary),
                strong_exports=283, providers=71, explicit_Runtime_calls=0, implicit_ldd_initialization="UNKNOWN")
    commands = [(f"{prefix}-{name}", hosts[name][key])
                for prefix, key in (("fresh", "preprocess"), ("compile", "compile")) for name in hosts]
    commands.append(("link", recipe["link"]))
    H.save(output / "CPU-result.json", dict(label=H.LABEL, request=H.REQUEST, step=NATIVE_ROW[0], node=H.NODE,
           claim_SHA256=H.CLAIM_PIN, actual=H.PLANNED, planned=H.PLANNED, **epochs, phase="complete",
           passed=True, completed=True, replay_allowed=False, GPU_GO=False, Model_GO=False, recipe_SHA256=pin,
           fresh_dependency_evidence=fresh, compile_dependency_evidence=actual, linked=link,
           commands=[dict(phase=phase, argv=argv, returncode=0, owned_child_reaped=True) for phase, argv in commands]))


def reject(function, expected):
    try:
        function()
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError(expected)


def main():
    _, waiter, _, _ = H.frozen()
    real_command = H.command
    tested = {}
    with TemporaryDirectory() as temp:
        for label in ("success", "unknown", "bad_kernel", "kernel_live", "foreign", "native_slot_busy",
                      "parent_end", "duplicate_step", "dependency_drift", "md_path_drift", "wide_cpu_mask"):
            root = Path(temp) / label
            (root / "results").mkdir(parents=True)
            (root / "candidates").mkdir()
            run_path = root / "results" / "controller"
            output = root / "candidates" / "worker"
            flow, commands = [], []
            row = NATIVE_ROW.copy()
            running = [*row[:2], "RUNNING", *row[3:]]
            replies = iter([row] if label == "duplicate_step" else [None] +
                           ([running]*6 if label == "unknown" else [running, row]))

            def kernel(known_claim, *, reconcile):
                assert known_claim == H.KERNEL_HOST_CLAIM and reconcile
                flow.append("kernel_terminal")
                got = proof()
                return dict(got, state="UNKNOWN") if label == "bad_kernel" else got

            def observe():
                flow.append("membership")
                active = KERNEL_ROW[0] if label == "kernel_live" else (
                    "1103606.650" if label == "native_slot_busy" and flow.count("membership") == 2 else None)
                return observed(active=active, foreign=(label == "foreign"))

            def expires():
                flow.append("parent_end")
                return time.time() + (10 if label == "parent_end" else 2100)

            def accounting(wait):
                assert 0 < wait <= 8
                flow.append("accounting")
                return next(replies)

            def wait(read, deadline, **kwargs):
                return waiter["wait_terminal"](read, deadline, interval=.001, **kwargs)

            def command(_, __, ___, epoch, pin, seconds):
                remote = PurePosixPath("/public/home/member/project")
                checkout = remote / "frozen-checkout"
                intent_path = remote / "results" / "native-controller"
                args = real_command(remote, checkout, intent_path, epoch, pin, seconds)
                assert f"{checkout}:{H.CHECKOUT}:ro" in args
                assert f"{intent_path / 'kernel-reconciliation.json'}:{H.PROOF}:ro" in args
                assert f"{remote / 'candidates' / H.KERNEL_LABEL}:/refactor/candidates/{H.KERNEL_LABEL}:ro" in args
                assert f"{remote / 'candidates'}:/refactor/candidates:rw" in args
                assert "--gres=none" in args and "--cpus-per-task=4" in args and "--mem=16384M" in args
                assert "--cpu-bind=threads" in args
                assert "--jobid=1103606" in args and "--nodelist=f12r2n13" in args
                assert f"--epoch {epoch} --deadline {epoch+1800} --work-deadline {epoch+1680}" in args[-1]
                assert "-I -S -B" in args[-1] and H.ENV.as_posix() in args[-1]
                if label == "success":
                    words = shlex.split(args[-1])
                    code = words[words.index("-c") + 1]
                    code = (
                        "import os;os.sched_getaffinity=lambda _: {2,4,6,8};"
                        "os.environ['SLURM_JOB_ID']='1103606';os.environ['SLURM_STEP_ID']='649';"
                        "os.environ['SLURM_CPUS_PER_TASK']='4'\n"
                    ) + code
                    result = subprocess.check_output([sys.executable, "-I", "-S", "-B", "-c", code,
                                                      str(H.REPO / H.WORKER), "--help"], timeout=7)
                    assert b"--kernel-reconciliation-sha256" in result
                    bind = json.loads(result.splitlines()[0])
                    assert bind["cpus"] == [2, 4, 6, 8] and bind["step"] == NATIVE_ROW[0]
                    wide = subprocess.run([sys.executable, "-I", "-S", "-B", "-c",
                                           code.replace("{2,4,6,8}", "set(range(8))", 1),
                                           str(H.REPO / H.WORKER), "--help"], capture_output=True, check=False, timeout=7)
                    assert wide.returncode != 0 and b"exactly four Slurm-bound CPUs" in wide.stderr
                    assert b"--kernel-reconciliation-sha256" not in wide.stdout
                return args

            def run(argv, *, timeout, stdout, **_):
                commands.append(argv)
                assert 0 < timeout < 1800
                intent = json.loads((run_path / "intent.json").read_bytes())
                kernel_pin = H.hash_bytes((run_path / "kernel-reconciliation.json").read_bytes())
                assert kernel_pin == intent["kernel_reconciliation_SHA256"]
                assert json.loads((run_path / "kernel-cleanup.json").read_bytes())["cleared_original_step"] == KERNEL_ROW[0]
                assert (run_path / "exec-attempt.json").exists()
                stdout.write((json.dumps(dict(kind=H.AFFINITY_KIND, step=row[0], cpus_per_task="4",
                    cpus=list(range(8 if label == "wide_cpu_mask" else 4)))) + "\n").encode())
                stdout.flush()
                if label == "unknown":
                    raise subprocess.TimeoutExpired(argv, 1)
                worker(output, intent, drift=(label == "dependency_drift"), file_drift=(label == "md_path_drift"))
                return SimpleNamespace(returncode=0)

            def launch():
                return H.launch(run_path, output, root/"checkout", reconcile_kernel=kernel, observe=observe,
                                expires=expires, accounting=accounting, runner=run, wait_terminal=wait)

            with patch.object(H, "command", side_effect=command):
                reasons = {"bad_kernel": "lacks", "kernel_live": "remains active", "foreign": "owner",
                           "native_slot_busy": "remains active", "parent_end": "expires", "duplicate_step": "already present"}
                if label in reasons:
                    reason = "no independently reconciled" if label == "bad_kernel" else reasons[label]
                    reject(launch, reason)
                    assert not commands and not run_path.exists()
                    if label == "bad_kernel":
                        assert flow == ["kernel_terminal"]
                    elif label in {"kernel_live", "foreign"}:
                        assert flow == ["kernel_terminal", "membership"]
                    tested[label] = "prelaunch_refused"
                    continue
                result = launch()
                assert flow[:5] == ["kernel_terminal", "membership", "membership", "parent_end", "accounting"]
                assert len(commands) == 1
                if label == "unknown":
                    assert result["state"] == "UNKNOWN" and result["client_timeout"] is True
                    reject(launch, "exists; reconcile")
                    retained = H.inspect_original(run_path, output, lambda timeout: running, waiter["is_terminal"])
                    assert retained["state"] == "UNKNOWN" and not output.exists()
                    worker(output, json.loads((run_path / "intent.json").read_bytes()))
                    resolved = H.inspect_original(run_path, output, lambda timeout: row, waiter["is_terminal"])
                    assert resolved["state"] == "NATIVE_VERIFIED" and len(commands) == 1
                elif label == "dependency_drift":
                    assert result["state"] == "TERMINAL_UNVERIFIED" and "dependency proof differs" in result["error"]
                elif label == "md_path_drift":
                    assert result["state"] == "TERMINAL_UNVERIFIED" and "M/MD paths differ" in result["error"]
                elif label == "wide_cpu_mask":
                    assert result["state"] == "TERMINAL_UNVERIFIED" and "exact four-CPU affinity" in result["error"]
                else:
                    assert result["state"] == "NATIVE_VERIFIED" and result["actual"] == H.PLANNED, result
                    original = (run_path / "accounting-result.json").read_bytes()
                    audit = H.inspect_original(run_path, output, lambda timeout: row, waiter["is_terminal"])
                    assert audit["library_SHA256"] == result["library_SHA256"]
                    assert (run_path / "accounting-result.json").read_bytes() == original
                    reject(lambda: H.inspect_original(run_path, output, lambda timeout: running, waiter["is_terminal"]),
                           "diverged")
                tested[label] = result["state"]

    line = ("|".join(NATIVE_ROW)+"\n").encode()
    assert H.accounting_row(line) == NATIVE_ROW
    reject(lambda: H.accounting_row(line+line), "ambiguous original")
    reject(lambda: H.accounting_row(line.replace(H.NODE.encode(), b"unowned")), "different parent or node")
    print(json.dumps(dict(fixtures=tested, duplicate_and_foreign_accounting_rejections=2,
                          actual_Slurm_calls=0, actual_native_compiles=0, actual_GPU_steps=0), indent=2))


if __name__ == "__main__":
    main()
