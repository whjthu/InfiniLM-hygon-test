"""Offline tests of the scope, handoff, and renderer. No Slurm commands are run."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_host
import host_scope as host


UID = 1234
NOW = 2000
BOOT = "b" * 64
PRE_ROW = ["1103606.2201", host.PRE_LABEL, "COMPLETED", "0:0", "1", host.NODE, "cpu=2,mem=4G,node=1"]
OLD_ROW = [host.OLD_STEP, host.OLD_LABEL, "COMPLETED", "0:0", "10500", host.NODE,
           "cpu=8,gres/dcu=8,mem=16G,node=1"]
GPU_ROW = ["1103606.2202", host.LABEL, "FAILED", "1:0", "80", host.NODE, "cpu=12,gres/dcu=8,mem=24G,node=1"]
POST_ROW = ["1103606.2203", host.POST_LABEL, "COMPLETED", "0:0", "2", host.NODE, "cpu=2,mem=4G,node=1"]


def inventory(request, step, start, end):
    record = host.encoded({"step": step, "node": host.NODE, "boot_id_SHA256": BOOT, "PCIs": sorted(host.PCI),
                           "kfd_process_count": 0, "own_device_fd_process_count": 0})
    return {"request": request, "stdout": record.decode(), "stdout_SHA256": host.digest(record),
            "source_SHA256": host.INVENTORY_SHA, "start_epoch": start, "observed_epoch": end}


def snapshot(records, epoch=NOW, numbered_steps=None):
    return {"parent": {"JobId": host.PARENT, "JobState": "RUNNING", "UserId": f"test({UID})",
                       "AllocTRES": "cpu=2560,gres/dcu=320,node=40"},
            "parent_nodes": [host.NODE, *[f"f13r1n{index:02}" for index in range(1, 40)]],
            "epoch": epoch, "accounting_columns": host.COLUMNS, "accounting_complete": True,
            "accounting_rows": deepcopy(records), "numbered_steps": numbered_steps or [],
            "numbered_steps_complete": True}


def admitted():
    return {"request": host.REQUEST, "label": host.LABEL, "transport_disposition": "NEVER_ATTEMPTED",
            "scheduler": snapshot([OLD_ROW, PRE_ROW]), "boot_id_SHA256": BOOT,
            "predecessor_terminal_observed_epoch": NOW - 15,
            "independent_pre": inventory(host.PRE_REQUEST, PRE_ROW[0], NOW - 12, NOW - 8)}


class ScopeTests(unittest.TestCase):
    def test_exact_gpu_argv_and_one_epoch(self):
        scope = host.validate_admission(admitted(), NOW, UID)
        sha = host.digest(host.encoded(scope))
        cmd = host.render_gpu(scope, sha, host.ATTESTED_SHA, host.DEVICE_HOST_SHA)
        self.assertEqual(scope["deadline_epoch"], NOW + 240)
        self.assertIn("--jobid=1103606", cmd)
        self.assertIn("--cpus-per-task=12", cmd)
        self.assertIn("--gres=dcu:8", cmd)
        self.assertIn("--time=00:04:00", cmd)
        self.assertIn("--cpu-bind=threads", cmd)
        self.assertIn("--scope-sha256", cmd)
        self.assertIn(sha, cmd)
        self.assertIn(str(host.STAGE / "device_host.py"), cmd)
        self.assertTrue(all(host.REQUEST_ID.fullmatch(item) for item in (host.REQUEST, host.PRE_REQUEST, host.POST_REQUEST)))
        with self.assertRaisesRegex(ValueError, "EXACT_GPU_RENDER_SCOPE"):
            host.render_gpu(scope, sha, host.ATTESTED_SHA, "a" * 64)

    def test_live_predecessor_stale_capture_and_unknown_attempt_are_refused(self):
        original = admitted()
        for mutate, reason in (
                (lambda d: d["scheduler"]["accounting_rows"][0].__setitem__(2, "RUNNING"),
                 "PREDECESSOR_STILL_PROTECTED"),
                (lambda d: d.__setitem__("transport_disposition", "UNKNOWN"), "ORIGINAL_REQUEST_STATE"),
                (lambda d: d["scheduler"].__setitem__("epoch", NOW - 31), "STALE_SCHEDULER"),
                (lambda d: d["scheduler"].__setitem__("accounting_complete", False), "COMPLETENESS"),
                (lambda d: d["scheduler"].__setitem__("numbered_steps_complete", False), "INCOMPLETE_NUMBERED"),
                (lambda d: d["scheduler"]["accounting_rows"].append(GPU_ROW), "ALREADY_VISIBLE"),
                (lambda d: d["scheduler"]["accounting_rows"].append(OLD_ROW), "EXACT_NUMBERED_ACCOUNTING"),
        ):
            with self.subTest(reason=reason):
                data = deepcopy(original)
                mutate(data)
                with self.assertRaisesRegex(ValueError, reason):
                    host.validate_admission(data, NOW, UID)

    def test_preflight_freshness_boot_and_holder_identity(self):
        for key, bad in (("kfd_process_count", 1), ("own_device_fd_process_count", False),
                         ("boot_id_SHA256", "f" * 64), ("PCIs", ["invalid"] * 8)):
            with self.subTest(key=key):
                data = admitted()
                result = data["independent_pre"]
                raw = json.loads(result["stdout"])
                raw[key] = bad
                result["stdout"] = host.encoded(raw).decode()
                result["stdout_SHA256"] = host.digest(result["stdout"].encode())
                with self.assertRaisesRegex(ValueError, "SAME_BOOT_EIGHT_PCI"):
                    host.validate_admission(data, NOW, UID)
        data = admitted()
        data["independent_pre"]["start_epoch"] = NOW - 16
        with self.assertRaisesRegex(ValueError, "CLEANUP_ORDER"):
            host.validate_admission(data, NOW, UID)

    def test_named_cgroup_owner_and_independent_post(self):
        scope = host.validate_admission(admitted(), NOW, UID)
        live = {"Name": host.LABEL, "NodeList": host.NODE, "State": "RUNNING", "StepId": GPU_ROW[0], "TRES": GPU_ROW[6]}
        receipt = {"request": host.REQUEST, "scope_SHA256": host.digest(host.encoded(scope)),
                   "entry_SHA256": host.ATTESTED_SHA, "runtime_SHA256": host.RUNTIME_SHA,
                   "slurm_listpids_matched": True, "live_slurm": live,
                   "owner": {"step": GPU_ROW[0], "node": host.NODE, "boot_SHA256": BOOT, "pid": 901,
                             "start_ticks": 1234, "epoch": NOW + 5, "affinity": list(range(12)), "uid": UID,
                             "cgroup_names_step": True, "cgroup_SHA256": "a" * 64}}
        gpu, cmd = host.post_after_terminal(snapshot([OLD_ROW, PRE_ROW, GPU_ROW]), scope, NOW, UID, attested=receipt)
        self.assertEqual(gpu, GPU_ROW)
        self.assertIn("--gres=none", cmd)
        self.assertIn("--cpus-per-task=2", cmd)
        host_raw = host.encoded(receipt)
        inside = {"request": host.REQUEST, "host_owner_SHA256": host.digest(host_raw),
                  "owner": dict(receipt["owner"], epoch=NOW + 6, pid=902, start_ticks=1235)}
        self.assertEqual(host.adopt_container_owner(host_raw, host.encoded(inside), scope)["step"], GPU_ROW[0])
        inside["owner"]["cgroup_SHA256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "CONTAINER_OWNER_DIFFERS"):
            host.adopt_container_owner(host_raw, host.encoded(inside), scope)
        with self.assertRaisesRegex(ValueError, "POST_ALREADY_ATTEMPTED"):
            host.post_after_terminal(snapshot([OLD_ROW, PRE_ROW, GPU_ROW, POST_ROW]), scope, NOW, UID, attested=receipt)
        receipt["owner"]["cgroup_names_step"] = False
        with self.assertRaisesRegex(ValueError, "ATTESTED_KERNEL_OWNER"):
            host.adopt_bound_owner(receipt, scope)

    def test_refill_requires_new_actual_step_with_all_eight_counters(self):
        scope = host.validate_admission(admitted(), NOW, UID)
        after = {"adopted_gpu_accounting": GPU_ROW, "gpu_terminal_observed_epoch": NOW + 241,
                 "independent_post": inventory(host.POST_REQUEST, POST_ROW[0], NOW + 242, NOW + 244)}
        clean_scheduler = snapshot([OLD_ROW, PRE_ROW, GPU_ROW, POST_ROW], NOW + 245)
        eligibility = host.refill_after_clean_post(clean_scheduler, scope, NOW + 245, UID, after)
        request = "b" * 32
        label = "training80-dcu-standby-" + request[:12] + "-" + host.NODE
        refill = ["1103606.2204", label, "RUNNING", "0:0", "12", host.NODE, OLD_ROW[-1]]
        current = snapshot([OLD_ROW, PRE_ROW, GPU_ROW, POST_ROW, refill], NOW + 260,
                           [{"StepId": refill[0], "Name": label, "State": "RUNNING", "nodes": [host.NODE]}])

        def point(epoch, calls):
            target_clock = epoch - 120
            return {"step": refill[0], "node": host.NODE, "boot_id_SHA256": BOOT, "epoch": epoch, "worker_PID": 700,
                    "worker_start_ticks": 900, "compute_observed_epoch": target_clock,
                    "rows": {pci: {"pci": pci, "active": True, "calls": calls, "epoch": target_clock - 1}
                             for pci in host.PCI}}

        evidence = {"request": request, "first": point(NOW + 250, 80), "second": point(NOW + 255, 120)}
        self.assertEqual(host.consume_refill(current, NOW + 260, UID, eligibility, evidence)["actual_computing_cards"], 8)
        evidence["second"]["rows"]["0000:09:00.0"]["calls"] = 80
        with self.assertRaisesRegex(ValueError, "REFILL_EIGHT_CARD_GEMM_GROWTH"):
            host.consume_refill(current, NOW + 260, UID, eligibility, evidence)

    def test_exclusive_attempt_path_is_reserved_before_return_and_refuses_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pre, run, post, stage = (root / leaf for leaf in ("pre", "gpu", "post", "stage"))
            stage.mkdir()
            far = Path(__file__).resolve().parent.parent / "zimage-transposed-lt-unseen-dispatch-v1"
            (stage / "node_inventory.py").write_bytes((far / "node_inventory.py").read_bytes())

            def publish(path, value):
                raw = host.encoded(value)
                with path.open("xb") as stream:
                    stream.write(raw)
                return host.digest(raw)

            with patch.multiple(host, PRE=pre, RUN=run, POST=post, STAGE=stage), \
                 patch.object(host, "require_lease") as guard, patch.object(host, "stage_pins"), \
                 patch.object(host, "save_new", side_effect=publish), \
                 patch.object(host.os, "getuid", return_value=UID, create=True), \
                 patch.object(host.time, "time", return_value=NOW - 15) as clock:
                lease = object()
                previous = snapshot([OLD_ROW], NOW - 15)
                _, intent_sha, cmd = host.prepare_pre_locked(lease, lambda: previous)
                self.assertIn("--gres=none", cmd)
                self.assertFalse(run.exists())
                clock.return_value = NOW
                data = admitted()
                data["independent_pre"]["intent_SHA256"] = intent_sha
                scope, _ = host.prepare_gpu_locked(lease, lambda: data, host.DEVICE_HOST_SHA)
                attempt = json.loads((run / "gpu-attempt.json").read_bytes())
                self.assertEqual(attempt["scope_SHA256"], host.digest((run / "scope.json").read_bytes()))
                self.assertEqual(attempt["admission_SHA256"], host.digest((run / "admission.json").read_bytes()))
                with self.assertRaisesRegex(ValueError, "NAMESPACE_ALREADY_USED"):
                    host.prepare_gpu_locked(lease, lambda: self.fail("no repeat collection"), host.DEVICE_HOST_SHA)
                clock.return_value = NOW + 241
                terminal = snapshot([OLD_ROW, PRE_ROW, GPU_ROW], NOW + 241)
                host.prepare_post_locked(lease, lambda: terminal, scope)
                with self.assertRaisesRegex(ValueError, "POST_NAMESPACE_ALREADY_USED"):
                    host.prepare_post_locked(lease, lambda: self.fail("no repeat collection"), scope)
                self.assertTrue(all(invocation.args == (lease,) for invocation in guard.call_args_list))

    def test_original_sources_and_same_process_cgroup_precheck(self):
        filename = Path(device_host.__file__).resolve()
        own = filename.read_bytes()
        self.assertEqual(host.digest(own), host.DEVICE_HOST_SHA)
        source = filename.parent.parent / "zimage-lt-eight-card-attested-entry-v1" / "node_entry.py"
        original = host.read_exact(source, host.ATTESTED_SHA)
        self.assertEqual(len(original), 9764)
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            launcher = stage / "device_host.py"
            launcher.write_bytes(own)
            (stage / "node_entry.py").write_bytes(original)
            data = host.encoded(host.validate_admission(admitted(), NOW, UID))
            (stage / "scope.json").write_bytes(data)
            args = SimpleNamespace(run=stage, entry_sha256=host.ATTESTED_SHA, launcher_sha256=host.DEVICE_HOST_SHA,
                                   scope_sha256=host.digest(data))
            called = []
            matched = False

            def bind(code, namespace):
                namespace["local_identity"] = lambda scope: {"cgroup_names_step": matched}
                namespace["host"] = lambda params: called.append(params) or 0

            with patch.object(device_host, "__file__", str(launcher)), patch.object(device_host, "exec", bind, create=True):
                with self.assertRaisesRegex(ValueError, "kernel cgroup"):
                    device_host.run(args)
                matched = True
                self.assertEqual(device_host.run(args), 0)
                self.assertEqual(called, [args])


if __name__ == "__main__":
    unittest.main()
