"""Exercise the overlay and fail-closed metadata join using recorded API shapes."""

from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare
import zimage_mm_operands as capture


class Event:
    def __init__(self, name, correlation, *, kind=capture.CPU, shapes=(), start=11, end=12, link=0, device=0):
        self.name = lambda: name
        self.correlation_id = lambda: correlation
        self.device_type = lambda: kind
        self.shapes = lambda: shapes
        self.start_ns = lambda: start
        self.end_ns = lambda: end
        self.linked_correlation_id = lambda: link
        self.device_index = lambda: device


def metadata(correlation, shapes, *, rhs_strides=None, device="cuda:0"):
    lhs, rhs = shapes
    inputs = [SimpleNamespace(sizes=shape, strides=stride, dtype="torch.bfloat16", layout="torch.strided",
                              device=device, storage_data_ptr=0xDEADBEEF)
              for shape, stride in [(lhs, (lhs[1], 1)), (rhs, rhs_strides or (1, rhs[0]))]]
    return SimpleNamespace(name="aten::mm", correlation_id=correlation, children=[],
                           extra_fields=SimpleNamespace(inputs=inputs))


def session(events, tree):
    raw = SimpleNamespace(events=lambda: events, experimental_event_tree=lambda: tree)
    return SimpleNamespace(profiler=SimpleNamespace(kineto_results=raw))


class Check(unittest.TestCase):
    def test_full_coverage_and_incomplete_signatures(self):
        events = [Event(capture.BACKWARD, 1, start=10, end=20)]
        tree = []
        correlation = 10
        for (m, n, k), count in capture.EXPECTED.items():
            for _ in range(count):
                correlation += 1
                shapes = [(m, k), (k, n)]
                events.extend([Event("aten::mm", correlation, shapes=shapes),
                               Event("Cijk_Alik_Bljk_ISA936_layout", 0, kind=capture.DEVICE, link=correlation)])
                tree.append(metadata(correlation, shapes))
        prof = session(events, tree)
        with TemporaryDirectory() as tmp:
            row = capture.write_capture(prof, Path(tmp), check=lambda: None)
            raw = (Path(tmp) / row["file"]).read_bytes()
            self.assertNotIn(b"storage_data_ptr", raw)
            self.assertEqual(row["SHA256"], sha256(raw).hexdigest())
            self.assertTrue(row["all_prior_linked_calls_covered"])
            self.assertEqual(row["shapes_with_bound_operands"], 3)
            with self.assertRaises(FileExistsError):
                capture.write_capture(prof, Path(tmp), check=lambda: None)
        tree[0].extra_fields.inputs[1].strides = []
        tree.append(tree[1])
        tree[3].extra_fields.inputs[0].device = "cuda:1"
        events[10] = Event("Cijk_Alik_Bljk_ISA936_layout", 0, kind=capture.DEVICE, link=0)
        unknown = capture.capture(prof, check=lambda: None)
        self.assertFalse(unknown["all_prior_linked_calls_covered"])
        total = unknown["totals"][0]
        self.assertEqual(total["missing_or_ambiguous_metadata"], 2)
        self.assertEqual(total["link_shape_or_device_mismatch"], 1)
        self.assertEqual(total["missing_cijk_link"], 1)
        self.assertEqual(total["metadata_joined"], 236)

    def test_actual_smoke_gate_and_source_overlay(self):
        shapes = [(64, 64)] * 2
        events = [Event("aten::mm", 21, shapes=shapes), Event("other", 22, kind=capture.DEVICE)]
        capture.verify_smoke(session(events, [metadata(21, shapes)]))
        with self.assertRaisesRegex(ValueError, "transposed"):
            capture.verify_smoke(session(events, [metadata(21, shapes, rhs_strides=(64, 1))]))
        sources = prepare.HERE.parent
        profile = sources / "training-zimage-resumed21-native-profile-v1/framework-delta" / prepare.PATH
        record = sources / "training-zimage-registered-hip-profile21-v1/framework-delta" / prepare.RECORD
        with TemporaryDirectory() as tmp:
            root, overlay = Path(tmp).resolve() / "framework", Path(tmp).resolve() / "overlay"
            for path, prior in [(prepare.PATH, profile), (prepare.RECORD, record)]:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(prior.read_bytes())
            receipt = prepare.stage(root, overlay)
            self.assertEqual(receipt["file_count"], 2)
            rendered = (overlay / prepare.PATH).read_bytes()
            self.assertLess(rendered.index(b"value=update"), rendered.index(b"=write_capture"))
            self.assertIn(b"record_shapes=True,profile_memory=False,with_stack=False", rendered)
            with self.assertRaises(FileExistsError):
                prepare.stage(root, overlay)
            (root / prepare.RECORD).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "framework member differs"):
                prepare.stage(root, overlay)


if __name__ == "__main__":
    unittest.main()
