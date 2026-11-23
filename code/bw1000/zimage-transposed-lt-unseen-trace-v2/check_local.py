"""Test the rendered tracer and exact fresh-standby gate without target access."""

import ast
from contextlib import nullcontext
import errno
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare


def need(ok, message):
    if not ok:
        raise ValueError(message)


class Entry:
    def __init__(self, name, size=0, mode=stat.S_IFREG, error=None):
        self.name, self.size, self.mode, self.error = name, size, mode, error
        self.stats = 0

    def stat(self, *, follow_symlinks):
        assert not follow_symlinks
        self.stats += 1
        if self.error:
            raise self.error
        return SimpleNamespace(st_size=self.size, st_mode=self.mode)


def functions(raw, names, globals_):
    all_source = ast.parse(raw)
    selected = ast.Module(body=[n for n in all_source.body if isinstance(n, ast.FunctionDef) and n.name in names],
                          type_ignores=[])
    exec(compile(selected, "<actual rendered scope>", "exec"), globals_)
    return globals_


def harness():
    shared = dict(os=os, spec=SimpleNamespace(need=need), Path=Path, stat=stat, time=time, signal=signal,
                  subprocess=subprocess, sys=sys, fingerprint=lambda *args: None,
                  TRACER=Path("/opt/dtk/hip/bin/hipprof"), TRACER_SHA="test_pin", TRACE_LIMIT=16 << 20)
    return functions(prepare.source_files()["worker.py"],
                     {"trace_size", "trace_error", "stop_trace_child", "in_trace"}, shared)


class Check(unittest.TestCase):
    def test_one_stat_per_member_and_strict_terminal(self):
        h = harness()
        vanished = Entry("temporary", error=FileNotFoundError(errno.ENOENT, "test"))
        stdout, stderr = Entry("tool.stdout.private", 8), Entry("tool.stderr.private", 3)
        h["os"] = SimpleNamespace(scandir=lambda _: nullcontext([stdout, vanished, stderr]))
        state = {"vanished_live_members": 0}
        measure = h["trace_size"]
        self.assertEqual(measure(Path("."), live=True, state=state), 11)
        self.assertEqual((stdout.stats, stderr.stats, vanished.stats), (1, 1, 1))
        self.assertEqual(state["vanished_live_members"], 1)
        with self.assertRaises(FileNotFoundError):
            measure(Path("."), live=False, state=state)
        h["os"].scandir = lambda _: nullcontext([stdout, stderr])
        self.assertEqual(measure(Path("."), live=False, state=state), 11)
        self.assertEqual((state["terminal_members"], state["terminal_bytes"]), (2, 11))
        for suspect, error in [(Entry("symlink", mode=stat.S_IFLNK), ValueError),
                               (Entry("illegal", error=PermissionError(errno.EACCES, "test")), PermissionError),
                               (Entry("oversize", size=(16 << 20) + 1), ValueError)]:
            h["os"].scandir = lambda _: nullcontext([stdout, suspect, stderr])
            with self.assertRaises(error):
                measure(Path("."), live=True, state=state)
        h["os"].scandir = lambda _: nullcontext([stderr])
        with self.assertRaisesRegex(ValueError, "MISSING_TERMINAL_TOOL_LOG"):
            measure(Path("."), live=False, state=state)

    def test_exec_live_and_terminal_errors_are_separated(self):
        with TemporaryDirectory() as tmp:
            for mode in ("exec", "live_scan", "terminal_scan"):
                h, directory, state = harness(), Path(tmp) / mode, {}
                directory.mkdir()
                calls = []
                child = SimpleNamespace(pid=1234, poll=lambda: None if mode == "live_scan" else 0)

                def launch(*_args, **_kwargs):
                    if mode == "exec":
                        raise FileNotFoundError(errno.ENOENT, "exec", str(h["TRACER"]))
                    return child

                def scan(_directory):
                    raise FileNotFoundError(errno.ENOENT, "scanner", str(directory))

                h["subprocess"] = SimpleNamespace(Popen=launch, DEVNULL=subprocess.DEVNULL,
                                                SubprocessError=subprocess.SubprocessError)
                h["os"] = SimpleNamespace(scandir=scan, fsdecode=os.fsdecode)
                h["stop_trace_child"] = lambda child: calls.append(child.pid)
                with self.assertRaises(FileNotFoundError):
                    h["in_trace"](["controlled_child"], directory, time.time() + 30, state)
                self.assertEqual(state["failure"]["phase"], "profiler_exec" if mode == "exec" else mode)
                self.assertEqual(state["failure"]["errno"], errno.ENOENT)
                self.assertEqual(state["child_PID"], None if mode == "exec" else 1234)
                self.assertEqual(calls, [1234] if mode == "live_scan" else [])
                self.assertEqual((directory / "tool.stdout.private").stat().st_size, 0)
                self.assertEqual((directory / "tool.stderr.private").stat().st_size, 0)

    def test_fresh_envelope_preserves_algorithm_and_rejects_old_authority(self):
        static = json.loads((prepare.HERE / "CLAIM.json").read_bytes())
        files = prepare.source_files()
        self.assertEqual(prepare.digest(files["worker.py"]), static["evidence"]["device_worker_SHA256"])
        prior = prepare.pinned(prepare.ORIGINAL / "worker.py", prepare.OLD_FILES["worker.py"])
        before = {n.name: ast.dump(n) for n in ast.parse(prior).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: ast.dump(n) for n in ast.parse(files["worker.py"]).body if isinstance(n, ast.FunctionDef)}
        for name in ("pci", "admission", "native_probe", "guarded", "fixed_inputs", "trace_api", "witness", "audit"):
            self.assertEqual(before[name], after[name], name)

        observed = prepare.HISTORICAL_EPOCH + 120
        cards = [prepare.PCI] + [f"0000:{index:02x}:00.0" for index in range(10, 17)]
        live = {prepare.STANDBY_NAME: dict(id=prepare.STANDBY, nodes="f12r2n11", state="RUNNING",
                                          tres="cpu=8,gres/dcu=8,mem=16G,node=1")}
        fixture = dict(epoch=observed, stable_8card_step=True, first_slurm=live, last_slurm=live, advancing=cards,
                       first=dict(step=prepare.STANDBY, observed_epoch=observed - 6,
                                  rows={pci: {"calls": 20} for pci in cards}),
                       second=dict(step=prepare.STANDBY, observed_epoch=observed - 1,
                                   rows={pci: {"calls": 21} for pci in cards}))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "fixture.json"
            snapshot.write_bytes(prepare.encode(fixture))
            value = prepare.fresh_handoff(snapshot, prepare.digest(snapshot.read_bytes()))
            staged = prepare.execution_files(value)
            prepare.stage(root / "rendered", staged)
            clock = SimpleNamespace(time=lambda: observed + 10)
            h = functions(staged["entry.py"], {"verify_fresh_handoff"}, {"require": need, "time": clock})
            claim = json.loads(staged["DISPATCH.json"])
            h["verify_fresh_handoff"](claim)
            clock.time = lambda: observed + 300
            with self.assertRaisesRegex(ValueError, "expired"):
                h["verify_fresh_handoff"](claim)
            fixture["epoch"] = prepare.HISTORICAL_EPOCH
            snapshot.write_bytes(prepare.encode(fixture))
            with self.assertRaisesRegex(ValueError, "fresh exact"):
                prepare.fresh_handoff(snapshot, prepare.digest(snapshot.read_bytes()))


if __name__ == "__main__":
    unittest.main()
