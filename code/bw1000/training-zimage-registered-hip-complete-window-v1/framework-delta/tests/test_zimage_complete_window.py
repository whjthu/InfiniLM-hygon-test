"""Check the native Z-Image loop's complete-window boundary without a device."""

import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation/src"))
from gencuda.zimage_native_record import run


def loop_code():
    function = ast.parse(inspect.getsource(run)).body[0]
    guarded = next(node for node in function.body if isinstance(node, ast.Try))
    loop_index = next(index for index, node in enumerate(guarded.body) if isinstance(node, ast.For)
                      and isinstance(node.target, ast.Name) and node.target.id == "step")
    # Include the proof emission after the real loop. Reuse the loop verbatim.
    initialization = guarded.body[loop_index - 2:loop_index]
    return compile(ast.Module(body=initialization + guarded.body[loop_index:loop_index + 2],
                              type_ignores=[]), "<zimage-native-window>", "exec")


class CompleteWindow(unittest.TestCase):
    def execute_loop(self, directory, *, fail_final_save=False):
        events = []
        clock = iter((1_000_000_000, 26_000_000_000))

        class Dataset:
            def __getitem__(self, index):
                events.append(("fetch", index))
                return index

        def update(*args, **kwargs):
            events.append(("update", args[3]))
            return {"loss": 0.125}

        def payload(**kwargs):
            step = kwargs["global_step"]
            events.append(("payload", step))
            return {"integrity_sha256": "a", "adapter_sha256": "b",
                    "metadata": {"next_data_cursor": step % 20}, "state": {}, "rng": {"cuda": {0: None}}}

        def save(path, value):
            step = int(path.parent.name.removeprefix("checkpoint-"))
            events.append(("save", step))
            if fail_final_save and step == 25:
                raise OSError("injected final checkpoint write error")
            path.parent.mkdir()
            path.write_bytes(b"checkpoint")
            events.append(("committed", step))

        def timestamp():
            stamp = next(clock)
            events.append(("clock", stamp))
            return stamp

        def sha(path):
            events.append(("audit_hash", int(path.parent.name.removeprefix("checkpoint-"))))
            return "a" * 64

        namespace = dict(
            output=directory, json=json, Path=Path, begin=0, stop=26, completed=[], checkpoints=[], metrics=[],
            measure_complete_window=True, profile_resumed_update=False, first_resumed_update_only=False,
            hip_retained=True, fixed_first=False, dataset=Dataset(),
            check=lambda: None,
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append(("warmup_fence", 5)))),
            time=SimpleNamespace(monotonic_ns=timestamp, monotonic=lambda: 0.1),
            recorded_native_update=update,
            common=SimpleNamespace(advance_optimizer_progress=lambda *args: None),
            resume=SimpleNamespace(make_payload=payload, save_atomic=save),
            model=SimpleNamespace(pipe=SimpleNamespace(scheduler=None)),
            sha=sha,
        )
        namespace.update(dict.fromkeys(("update", "native_module", "recorder", "optimizer",
                                         "optimizer_progress", "scheduler", "contract", "update_reference")))
        if fail_final_save:
            with self.assertRaises(OSError):
                exec(loop_code(), namespace)
        else:
            exec(loop_code(), namespace)
        return events, namespace

    def test_includes_fetch_and_durable_saves_and_excludes_audit_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, result = self.execute_loop(Path(tmp))
            start = events.index(("clock", 1_000_000_000))
            end = events.index(("clock", 26_000_000_000))
            self.assertEqual(events[start - 2:start + 2],
                             [("update", 5), ("warmup_fence", 5), ("clock", 1_000_000_000), ("fetch", 5)])
            self.assertEqual(events[end - 1], ("committed", 25))
            self.assertEqual([event[1] for event in events[start:end] if event[0] == "update"], list(range(6, 26)))
            self.assertEqual([event[1] for event in events[start:end] if event[0] == "committed"], [10, 20, 25])
            self.assertEqual(events[end + 1:], [("audit_hash", step) for step in (10, 20, 25)])
            self.assertEqual(result["complete_window"]["updates_per_second"], 0.8)
            self.assertEqual(len((Path(tmp) / "steps.private.jsonl").read_text().splitlines()), 25)

    def test_final_save_error_never_emits_a_complete_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, result = self.execute_loop(Path(tmp), fail_final_save=True)
            self.assertEqual(result["completed"], list(range(1, 26)))
            self.assertIsNone(result["complete_window"])
            self.assertNotIn(("clock", 26_000_000_000), events)
            self.assertFalse((Path(tmp) / "steps.private.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
