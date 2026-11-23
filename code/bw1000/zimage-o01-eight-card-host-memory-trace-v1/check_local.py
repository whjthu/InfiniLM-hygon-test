"""Test the readback interlock and last-sample preservation off cluster."""

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ram_trace as t


STEP, SIDE = "1103606.2300", "1103606.2301"
READER = Path(__file__).resolve().parent.parent / "zimage-o01-standby2179-host-memory-observer-v1/observer.py"


def scope(epoch=None):
    epoch = time.time() - 5 if epoch is None else epoch
    return {"request": "a" * 32, "label": "fixture-next-gpu", "parent": t.PARENT, "node": t.NODE,
            "allocated_DCU": 8, "epoch": epoch, "deadline_epoch": epoch + 240, "runtime_SHA256": "b" * 64,
            "attested_entry_SHA256": "c" * 64, "boot_id_SHA256": "d" * 64, "host_memory_MiB": t.PROPOSED_MIB,
            "memory_observer_label": "fixture-next-observer"}


def body(data, filename, content):
    (data / filename).write_text(content + "\n", encoding="ascii")


class TraceTest(unittest.TestCase):
    def test_reserved_budget_and_exact_accounted_48G(self):
        bound = scope()
        limit = 48 << 30
        self.assertEqual(t.scope_checks(bound), limit)
        row = f"StepId={STEP} Name={bound['label']} NodeList={t.NODE} State=RUNNING TRES=cpu=12,gres/dcu=8,mem=48G,node=1"
        t.inspect_resources("StepId=other\n" + row, bound, STEP, bound["label"], 12, 8, 49152)
        for bad in (row.replace("48G", "24G"), row + "\n" + row, row.replace("cpu=12", "cpu=11")):
            with self.assertRaises(ValueError):
                t.inspect_resources(bad, bound, STEP, bound["label"], 12, 8, 49152)
        with self.assertRaisesRegex(ValueError, "ANCESTOR_LIMIT"):
            t.budget({"step_memory": {"limit_bytes": {"value": limit}}, "minimum_visible_hard_limit_bytes": limit - 1},
                     limit)
        for change in ({"host_memory_MiB": 24576}, {"label": t.OLD_GPU_LABEL},
                       {"memory_observer_label": t.OLD_OBSERVER_LABEL}, {"request": t.OLD_GPU_REQUEST}):
            with self.assertRaises(ValueError):
                t.scope_checks({**bound, **change})

    def test_phase_commit_never_overwrites_and_ready_binds_step(self):
        data = scope()
        sha = t.digest(t.encoded(data))
        with tempfile.TemporaryDirectory() as folder, patch.object(t, "sync_directory"):
            root = Path(folder)
            t.record_phase(root, data, sha, STEP, "cold_standbys_enter")
            phase = t.phase_path(root, "cold_standbys_enter")
            content = phase.read_bytes()
            with self.assertRaises(FileExistsError):
                t.record_phase(root, data, sha, STEP, "cold_standbys_enter")
            self.assertEqual(phase.read_bytes(), content)
            t.record_phase(root, data, sha, STEP, "witness_enter")
            self.assertEqual(t.latest_phase(root, data, sha, STEP)["phase"], "witness_enter")
            with self.assertRaisesRegex(ValueError, "PHASE_RECEIPT_IDENTITY"):
                t.latest_phase(root, data, sha, SIDE)
            p = root / "ready.json"
            t.commit(p, {"request": data["request"], "step": STEP, "scope_SHA256": sha,
                         "actual_host_limit_bytes": 48 << 30, "monotonic_ns": time.monotonic_ns(),
                         "first_sample_SHA256": "e" * 64})
            self.assertEqual(t.await_baseline(p, data, sha, STEP)["actual_host_limit_bytes"], 48 << 30)
            with self.assertRaisesRegex(ValueError, "BASELINE_READY_IDENTITY"):
                t.await_baseline(p, data, sha, SIDE)

    def test_v1_reader_and_fsynced_first_sample_survive_cgroup_disappearance(self):
        old = t.memory_module(READER)
        data = scope()
        limit = 48 << 30
        sha = t.digest(t.encoded(data))
        observer_row = (f"StepId={SIDE} Name={data['memory_observer_label']} NodeList={t.NODE} "
                        "State=RUNNING TRES=cpu=1,mem=256M,node=1")
        gpu_row = (f"StepId={STEP} Name={data['label']} NodeList={t.NODE} "
                   "State=RUNNING TRES=cpu=12,gres/dcu=8,mem=48G,node=1")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "cgroup"
            path.mkdir()
            for filename, val in {"memory.limit_in_bytes": str(limit), "memory.usage_in_bytes": "10000",
                                  "memory.max_usage_in_bytes": "12000", "memory.failcnt": "2",
                                  "memory.oom_control": "under_oom 0\noom_kill 1", "memory.stat": "rss 9999"}.items():
                body(path, filename, val)
            out = root / "observer"
            out.mkdir()
            args = SimpleNamespace(reader=READER, attested_source=root / "attested", owner=root / "owner",
                                   phases=root / "unstarted_phases", out=out, scope_sha256=sha)
            real_sample = old.memory_counters(root, path, 1)
            reader = SimpleNamespace(load_identity=lambda _: None, process_binding=lambda *args: None,
                                     control=lambda *args: f"{observer_row}\n{gpu_row}")
            calls = []

            def counters(*args):
                calls.append(args)
                if len(calls) == 1:
                    return real_sample
                raise FileNotFoundError("fixture cgroup already removed")

            reader.memory_counters = counters
            with patch.object(t, "memory_module", return_value=reader), patch.object(t, "await_owner", return_value={}), \
                 patch.object(t, "target_binding", return_value=(STEP, 1, root, path)), \
                 patch.object(t.os, "sched_getaffinity", return_value={99}, create=True), \
                 patch.object(t.socket, "gethostname", return_value=t.NODE), patch.object(t.time, "sleep"), \
                 patch.object(t, "sync_directory"), patch.dict(t.os.environ, {"SLURM_JOB_ID": t.PARENT,
                     "SLURM_STEP_ID": SIDE.split(".")[1], "SLURM_CPUS_PER_TASK": "1"}, clear=True):
                result = t.run(args, data, limit)
            self.assertEqual(result["end_reason"], "ORIGINAL_CGROUP_OR_COUNTER_DISAPPEARED")
            self.assertEqual(result["samples"], 1)
            self.assertFalse(result["step_terminal_independent"])
            raw = (out / "samples.jsonl").read_bytes()
            ready = json.loads((out / "ready.json").read_bytes())
            self.assertEqual(ready["first_sample_SHA256"], t.digest(raw))
            self.assertEqual(json.loads(raw)["memory"]["step_memory"]["oom_control"]["value"]["oom_kill"], 1)

    def test_v2_keeps_event_fields_and_detects_inode_replacement(self):
        actual = t.memory_module(READER)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            group = root / "fake-step"
            group.mkdir()
            cap = 48 << 30
            for filename, val in {"memory.max": str(cap), "memory.current": "3000", "memory.peak": "5000",
                                  "memory.events": "oom 2\noom_kill 1", "memory.stat": "anon 2800"}.items():
                body(group, filename, val)
            node = group.stat()
            row = t.sample(actual, root, group, 2, (node.st_dev, node.st_ino))
            self.assertEqual(t.budget(row, cap), cap)
            self.assertEqual(row["step_memory"]["events"]["value"], {"oom": 2, "oom_kill": 1})
            with self.assertRaisesRegex(ValueError, "TARGET_CGROUP_REPLACED"):
                t.sample(actual, root, group, 2, (node.st_dev, node.st_ino + 1))


if __name__ == "__main__":
    unittest.main()
