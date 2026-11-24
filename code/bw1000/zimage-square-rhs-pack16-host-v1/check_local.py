"""Local-only verification of the Pack16 HOST command and failure gates."""

import copy
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

import host
import worker_gate


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
LOCAL = ROOT / "local-only/O01-pack16-preflight-v3-c0243062d1f24b70aee7e3092c24e8f0"
EPOCH = 1795500000
FIXTURE_NODE = "f99r9n99"
BOOT = "00000000-0000-4000-8000-000000000001"
FIXTURE_ROOT = host.HOST_ROOT / "fixture-only-pack16-host"
PCI_LIST = [f"0000:{slot:02x}:00.0" for slot in range(8)]


def store(path, value):
    raw = host.encode(value)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def rejected(action, expected):
    try:
        action()
    except ValueError as error:
        assert str(error) == expected
    else:
        raise AssertionError("failed-closed gate unexpectedly accepted: " + expected)


def fixtures(path):
    step = "1103606.901"
    pre = {"phase": "pre", "observed_epoch": EPOCH - 3,
           "accounting": [step, "fixture-pre", "COMPLETED", "0:0", "2", FIXTURE_NODE, "cpu=1,mem=256M,node=1"],
           "inventory": {"node": FIXTURE_NODE, "step": step,
                         "boot_id_SHA256": sha256((BOOT + "\n").encode()).hexdigest(),
                         "PCIs": PCI_LIST, "kfd_process_count": 0, "own_device_fd_process_count": 0},
           "assigned_PCI": PCI_LIST[0], "gfx": "gfx936", "exclusive_1_DCU_lease": True}
    guard = {"parent": host.PARENT, "protected_nodes": ["f12r2n11", "f12r2n10"],
             "protected_steps": ["1103606.2559", "1103606.2615"], "selected_node": FIXTURE_NODE,
             "observed_epoch": EPOCH - 4, "valid_until_epoch": EPOCH + 60}
    scope = {"request": host.REQUEST, "label": host.LABEL, "source_publication": host.PUBLICATION,
             "parent": host.PARENT, "node": FIXTURE_NODE, "PCI": PCI_LIST[0], "boot_id": BOOT,
             "pre_step": step, "root_admitted": True, "gfx": "gfx936", "torch_version": "2.10.0",
             "torch_hip": "6.3.26113", "runtime_SHA256": host.RUNTIME_SHA, "epoch": EPOCH,
             "deadline_epoch": EPOCH + 240, "protected_nodes": guard["protected_nodes"],
             "pre_SHA256": store(path / "pre.json", pre),
             "protection_authority_SHA256": store(path / "protection.json", guard)}
    scope_sha = store(path / "scope.json", scope)
    return pre, guard, scope, scope_sha


def argv_capture(path, scope_sha):
    cli = [sys.executable, "-B", "-X", "utf8", str(BASE / "host.py"), "dry-run",
           "--repo", str(ROOT), "--runtime", str(LOCAL / "runtime"),
           "--original-dependencies", str(LOCAL / "original-outcome.json"),
           "--scope", str(path / "scope.json"), "--pre", str(path / "pre.json"),
           "--protection", str(path / "protection.json"), "--scope-sha256", scope_sha,
           "--target-runtime", str(FIXTURE_ROOT / "runtime"), "--target-scope", str(FIXTURE_ROOT / "scope.json"),
           "--target-output", str(FIXTURE_ROOT / "output"), "--observed-now", str(EPOCH + 1)]
    completed = subprocess.run(cli, cwd=path, text=True, capture_output=True, timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr
    view = json.loads(completed.stdout)
    assert view["dry_run"] and view["target_calls"] == 0
    argv = view["argv"]
    assert argv.count("srun") == 1 and argv.count("--gres=dcu:1") == 1 and "--overlap" not in argv
    assert {"--jobid=1103606", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=8192M",
            "--time=00:04:00", "--cpu-bind=threads", "--kill-on-bad-exit=1"} <= set(argv)
    assert sum(item.endswith(":rw") for item in argv) == 1 and argv.count("--bind") == 5
    assert host.HOST_PYTHON in argv and str(FIXTURE_ROOT / "worker_gate.py") in argv
    assert argv.index(host.HOST_PYTHON) < argv.index(host.APPTAINER)
    assert "exec /usr/bin/timeout --kill-after=5s" in argv[-1]
    assert "/pair-package/worker.py --scope /pack16-scope.json" in argv[-1] and "--cell" not in argv[-1]
    assert view["planned"] == host.DENOMINATOR and view["client_timeout_seconds"] == 205
    return view


def test_guards(scope, guard, pre, scope_sha):
    paths = [str(FIXTURE_ROOT / name) for name in ("runtime", "scope.json", "output")]
    for delta, reason, now in (
        ({"node": "f12r2n11"}, "PHYSICAL_SCOPE", EPOCH),
        ({"node": "f12r2n10"}, "PHYSICAL_SCOPE", EPOCH),
        ({"pre_step": "1103606.2662"}, "FRESH_AUTHORITY", EPOCH),
        ({"deadline_epoch": EPOCH + 241}, "ORIGINAL_EPOCH", EPOCH),
        ({}, "ORIGINAL_EPOCH", EPOCH + 30),
    ):
        rejected(lambda: host.build_argv(dict(scope, **delta), scope_sha, *paths, now), reason)
    rejected(lambda: host.build_argv(scope, scope_sha, paths[0], paths[1], paths[0], EPOCH), "WORKSPACE_ISOLATION")
    rejected(lambda: host.protection_gate(dict(guard, valid_until_epoch=EPOCH), scope, EPOCH),
             "PROTECTED_WORK_OR_EXPIRED_AUTHORITY")
    dirty = copy.deepcopy(pre)
    dirty["inventory"]["kfd_process_count"] = 1
    rejected(lambda: host.clean_inventory(dirty, scope, "pre", EPOCH), "INDEPENDENT_CLEAN_INVENTORY")
    boot_moved = copy.deepcopy(pre)
    boot_moved["inventory"]["boot_id_SHA256"] = "0" * 64
    rejected(lambda: host.clean_inventory(boot_moved, scope, "pre", EPOCH), "INDEPENDENT_CLEAN_INVENTORY")


def test_scheduler(scope):
    nodes = [FIXTURE_NODE] + [f"f98r9n{i:02d}" for i in range(39)]
    def observed(argv, _deadline):
        if argv[:3] == ["scontrol", "show", "job"]:
            return [f"JobId=1103606 JobState=RUNNING UserId=fixture(123) NodeList={FIXTURE_NODE} AllocTRES=gres/dcu=320"]
        if argv[:3] == ["scontrol", "show", "hostnames"]:
            return nodes
        if argv[:3] == ["scontrol", "show", "node"]:
            return [f"NodeName={FIXTURE_NODE} CPUTot=64 RealMemory=65536 State=ALLOCATED"]
        if argv[:3] == ["scontrol", "show", "step"]:
            return [f"StepId=1103606.902 Name=unrelated State=RUNNING NodeList={FIXTURE_NODE} TRES=gres/dcu=1"]
        raise AssertionError("unexpected query")
    watcher = host.helper(ROOT, host.ACCOUNT_HELPER)
    with mock.patch.object(host.os, "getuid", return_value=123, create=True), mock.patch.object(host, "query", side_effect=observed):
        rejected(lambda: host.idle_owned(scope, 1, watcher.TERMINAL), "CANDIDATE_NODE_OCCUPIED")
    running = ["1103606.902", host.LABEL, "RUNNING", "0:0", "1", FIXTURE_NODE, "cpu=4,gres/dcu=1,mem=8G,node=1"]
    outcome = watcher.wait_terminal(lambda: running, 10, max_observations=2, monotonic=lambda: 0, sleep=lambda _: None)
    assert not outcome["terminal_visible"] and outcome["observations"] == [running, running]


def test_startup(scope):
    state = {"held": False}
    child = mock.Mock()
    child.poll.return_value = None

    @contextmanager
    def fake_lock():
        state["held"] = True
        try:
            yield
        finally:
            state["held"] = False

    def prepare():
        assert state["held"]
        return ["fixture"], 20

    def spawn(argv):
        assert state["held"] and argv == ["fixture"]
        return child

    for expected in ("1103606.902", None):
        def observe(_deadline):
            assert state["held"]
            return expected

        owned, deadline, numeric, error = host.submit_under_lock(
            fake_lock(), prepare, spawn, observe, clock=lambda: 0, sleep=lambda _: None)
        # The caller's long reaper and terminal polling can only run after this return.
        assert not state["held"] and (owned, deadline, numeric, error) == (child, 20, expected, None)
    with mock.patch.object(host, "query", return_value=[
            "StepId=1103606.902 Name=" + host.LABEL + " State=RUNNING NodeList=" + FIXTURE_NODE
            + " TRES=cpu=4,gres/dcu=1,node=1,mem=8G"]):
        assert host.visible_step(scope, 4) == "1103606.902"

    calendar = ROOT / "local-only/training80-0544-ordinary-70-exact-predecessor-calendar-preparation-v1.json"
    rejected(lambda: host.ordinary_exclusion(ROOT, calendar, "f12r2n10"),
             "EXACT_ORDINARY_REFILL_EXCLUSION_UNAVAILABLE")
    assert host.pin((BASE / "worker_gate.py").read_bytes())["SHA256"] == host.GATE_SHA
    env = {"SLURM_JOB_ID": host.PARENT, "SLURM_STEP_ID": "902", "SLURM_CPUS_PER_TASK": "4"}
    prefix = "StepId=1103606.902 Name=" + host.LABEL + " State=RUNNING NodeList=" + FIXTURE_NODE + " TRES="
    def at_worker(args):
        assert args == ["scontrol", "show", "step", "1103606.902", "--oneliner"]
        return prefix + "cpu=4,gres/dcu=1,mem=8192M,node=1"
    assert worker_gate.exact_step(scope, env, FIXTURE_NODE, at_worker) == "1103606.902"
    for tres in ("cpu=4,gres/dcu=8,mem=8192M,node=1", "cpu=4,mem=8192M,node=1",
                 "cpu=4,gres/dcu=1,mem=16384M,node=1", "cpu=8,gres/dcu=1,mem=8192M,node=1"):
        rejected(lambda: worker_gate.exact_step(scope, env, FIXTURE_NODE, lambda _: prefix + tres),
                 "PRE_TORCH_EXACT_TRES")
    rejected(lambda: worker_gate.exact_step(scope, dict(env, SLURM_STEP_ID="batch"), FIXTURE_NODE, at_worker),
             "PRE_TORCH_NUMERIC_STEP")


def test_finish(path, pre, scope, scope_sha):
    step = "1103606.902"
    data = [step, host.LABEL, "FAILED", "1:0", "2", FIXTURE_NODE, "cpu=4,gres/dcu=1,mem=8G,node=1"]
    report = {"request": host.REQUEST, "scope_SHA256": scope_sha, "state": "TERMINAL", "actual_resources_match": True,
              "accounting_columns": list(host.COLUMNS), "accounting": [data], "terminal_observed_epoch": EPOCH + 3,
              "owned_srun_rc": 1, "forced_client_reap": False, "log_within_bound": True}
    poststep = "1103606.903"
    post = {"phase": "post", "observed_epoch": EPOCH + 4,
            "inventory": dict(pre["inventory"], step=poststep),
            "accounting": [poststep, "fixture-post", "COMPLETED", "0:0", "2", FIXTURE_NODE, "cpu=1,mem=256M,node=1"],
            "after_gpu_step": step, "after_terminal": True, "terminal_row_SHA256": host.pin(host.encode(data))["SHA256"],
            "independent_gpu_accounting": data}
    out = path / "test-output"
    out.mkdir()
    host_receipt = out / "host-terminal.json"
    host_sha = store(host_receipt, report)
    def finish(proof, terminal=report):
        return host.independently_finish(ROOT, scope, scope_sha, pre, proof, terminal,
                                        host_receipt, host_sha, LOCAL / "runtime", path / "scope.json", out)
    allowed = finish(post)
    assert allowed["planned"] == host.DENOMINATOR and allowed["independent_post"] == "CLEAN"
    assert not allowed["component_timing_candidate"] and not allowed["promotion_allowed"]
    rejected(lambda: finish(dict(post, observed_epoch=EPOCH + 2)), "POST_BEFORE_TERMINAL")
    rejected(lambda: finish(dict(post, independent_gpu_accounting=list(reversed(data)))), "INDEPENDENT_GPU_ACCOUNTING_DIFFERS")
    rejected(lambda: finish(post, dict(report, state="UNKNOWN")), "HOST_NOT_INDEPENDENTLY_FINALIZABLE")
    with mock.patch.object(host, "query", return_value=["|".join(data)] * 2):
        rejected(lambda: host.accounting_row(1), "AMBIGUOUS_ORIGINAL_LABEL")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("expected one absent private receipt path")
    with tempfile.TemporaryDirectory(prefix="pack16-host-dry-") as temp:
        path = Path(temp).resolve(strict=True)
        pre, guard, scope, sha = fixtures(path)
        capture = argv_capture(path, sha)
        test_guards(scope, guard, pre, sha)
        test_scheduler(scope)
        test_startup(scope)
        test_finish(path, pre, scope, sha)
    receipt = {
        "schema": "o01-pack16-host-local-verification-v1", "source_SHA256": host.pin((BASE / "host.py").read_bytes())["SHA256"],
        "captured_fixture_argv": capture["argv"], "fixture_node": FIXTURE_NODE, "planned": host.DENOMINATOR,
        "checks": ["frozen actual 2662 and 5-file runtime", "single exact argv", "protected nodes and epoch rejects",
                   "clean inventory and active-numbered-step rejection", "running/duplicate accounting rejection",
                   "failed denominator and independent post binding",
                   "actual ordinary calendar excludes no f12r2n10 run",
                   "start lock released on visible and unknown ownership", "pre-Torch numeric step and exact TRES"],
        "deployment_state": "LOCAL_ONLY_NO_ADOPTED_F12R2N10_REFILL_EXCLUSION",
        "historical": capture["sources"], "local_target_calls": 0, "actual_GPU_steps": 0, "new_Slurm_calls": 0,
    }
    print(json.dumps(host.exclusive(Path(sys.argv[1]), receipt), sort_keys=True))


if __name__ == "__main__":
    main()
