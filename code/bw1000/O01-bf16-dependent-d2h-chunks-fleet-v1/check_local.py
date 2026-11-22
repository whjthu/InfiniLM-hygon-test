"""Verify isolated staged transfer execution, accounting, and actual queue admission."""

import ctypes as C
from pathlib import Path
import runpy
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import derive
import launch
import package
from contract import ACTIVE_SECONDS, NANO, ORDER, TRANSFER_BYTES, counters, grid
from controller import unpack
from scope import EXCLUDED_NODES, MAX_NODES, PRIORITY_NODES, REQUEST
import test_report_training80_backlog as dispatch_fixture


class FakeHIP:
    D2H, H2D = 2, 1
    GUARD = 256
    GUARDS = (b"\xa5" * GUARD, b"\x5a" * GUARD)

    def __init__(self):
        self.allocations = []
        self.calls = []

    @staticmethod
    def add(ptr, offset):
        return C.c_void_p(ptr.value + offset)

    def new_buffer(self, hip, resources, host, nbytes):
        raw = C.create_string_buffer(nbytes + 2 * self.GUARD)
        self.allocations.append(raw)
        base = C.c_void_p(C.addressof(raw))
        return base, self.add(base, self.GUARD), self.add(base, self.GUARD + nbytes)

    def call(self, operation, *args):
        if operation == "MemcpyAsync":
            dst, src, size, direction, stream = args
            self.calls.append((operation, dst.value, src.value, size, direction, stream))
            C.memmove(dst, src, size)
        elif operation == "Memcpy":
            dst, src, size, _ = args
            C.memmove(dst, src, size)
        else:
            self.calls.append((operation, *args))


class NumericSplit(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        staged = Path(work.name)
        _, files = package.source(package.REPO)
        for name, raw in files.items():
            (staged / name).write_bytes(raw)
        self.module = runpy.run_path(str(staged / "worker.py"))

    def test_reproduce_frozen_artifacts(self):
        for name, raw in derive.render(derive.REPO).items():
            self.assertEqual((HERE / name).read_bytes(), raw)

    def test_real_worker_only_splits_async_request_at_the_midpoint(self):
        hip = FakeHIP()
        transfer = self.module["Transfer"](hip, hip, None, 3119)
        transfer.prepare_chunks()
        timed_unit = self.module["timed_unit"]
        clock = SimpleNamespace(monotonic=iter(range(100)).__next__)
        replays = tuple(lambda index=index: hip.calls.append(("graph", index)) for index in range(2))
        compute = ("compute", "compute_first", "compute_last")
        copy = ("same_copy_stream", "copy_first", "copy_last")
        outputs = []
        with patch.object(timed_unit.__globals__["prior"].dma, "elapsed_ms", return_value=1.0):
            for arm in ("one_copy", "two_chunks", "common_one_copy"):
                transfer.poison()
                hip.calls.clear()
                timed_unit(hip, compute, copy, replays, transfer, arm, clock)
                outputs.append(hip.calls[:])
                self.assertEqual(transfer.audit(), transfer.pin)
        self.assertEqual(outputs[0], outputs[2])
        extracts = [[call for call in arm if call[0] == "MemcpyAsync"] for arm in outputs]
        unchanged = [[call for call in arm if call[0] != "MemcpyAsync"] for arm in outputs]
        self.assertEqual(unchanged[0], unchanged[1])
        original, split, _ = extracts
        self.assertEqual((len(original), len(split)), (1, 2))
        half = TRANSFER_BYTES // 2
        _, dest, src, size, direction, stream = original[0]
        self.assertEqual(size, TRANSFER_BYTES)
        self.assertEqual(split, [("MemcpyAsync", dest, src, half, direction, stream),
                                 ("MemcpyAsync", dest+half, src+half, half, direction, stream)])
        for trace in outputs[:2]:
            first = trace.index(("EventRecord", "copy_first", stream))
            last = trace.index(("EventRecord", "copy_last", stream))
            self.assertEqual(trace[first+1:last], [x for x in trace if x[0] == "MemcpyAsync"])
            self.assertGreater(trace.index(("StreamSynchronize", "compute")), last)
            self.assertEqual([x for x in trace if x[0] == "StreamSynchronize"],
                             [("StreamSynchronize", "compute"), ("StreamSynchronize", stream)])

    def test_missing_second_chunk_and_corrupted_source_are_rejected(self):
        hip = FakeHIP()
        transfer = self.module["Transfer"](hip, hip, None, 3119)
        transfer.prepare_chunks()
        transfer.poison()
        dest, source, half = transfer.placements[1][0]
        hip.call("MemcpyAsync", dest, source, half, hip.D2H, "same")
        with self.assertRaisesRegex(ValueError, "D2H payload SHA"):
            transfer.audit()
        transfer.enqueue("same", 2)
        C.memset(transfer.device[1], 0, TRANSFER_BYTES)
        with self.assertRaisesRegex(ValueError, "source payload"):
            transfer.audit()
        C.memset(transfer.device[2], 0, 1)
        transfer.pin = derive.sha(b"\0" * TRANSFER_BYTES)
        transfer.enqueue("same", 2)
        with self.assertRaisesRegex(ValueError, "redzone"):
            transfer.audit()

    def test_closure_counts_only_copy_calls_as_the_work_difference(self):
        one = counters("one_copy", 19)
        two = counters("two_chunks", 19)
        self.assertEqual((one.pop("transfer_calls"), two.pop("transfer_calls")), (19, 38))
        self.assertEqual(one, two)
        self.assertEqual((one["d2h_payload_bytes"], one["stream_fences"], one["projection_calls"]),
                         (19 * TRANSFER_BYTES, 38, 19 * 36))

    def test_packed_finite_source_and_shared_queue_respect_fresh_and_protected(self):
        fleet = dispatch_fixture.BacklogTest()
        fleet.setUp()
        self.addCleanup(fleet.doCleanups)
        fleet.now = time.monotonic()
        fleet.anchor = time.time() - fleet.now
        for item in fleet.snapshot["nodes"].values():
            item.update(PCIs=[f"0000:{value:02x}:00.0" for value in range(1, 9)],
                        device_probe_step=f"{item['parent']}.900", kfd_process_count=0,
                        own_device_fd_process_count=0, probe_error=None)
        snapshot = fleet.verified_snapshot()
        protected = {"f12r2n13"}
        campaign, body, members = package.prepare("a"*40, 1000.0, snapshot, protected)
        self.assertIn(package.TRANSFER_BASELINE.as_posix(), members)
        self.assertEqual(campaign["request"], REQUEST)
        self.assertEqual(set(campaign["selected"]), set(snapshot["nodes"]) - EXCLUDED_NODES - PRIORITY_NODES - protected)
        self.assertLessEqual(len(body), 131072)
        self.assertEqual(len(campaign["selected"]), MAX_NODES-1)
        self.assertTrue(all(len(unit["nodes"]) <= 4 for unit in campaign["shards"].values()))
        for index in range(3):
            self.assertEqual([(mode, (end-start)//NANO) for mode, start, end in grid(campaign, index)],
                             [(arm, ACTIVE_SECONDS) for arm in ORDER])
        compiled = compile(body, "finite source", "exec")
        self.assertIsNotNone(compiled)
        with patch("sys.argv", ["controller.py"]):
            header = body.partition(b"\n")[0]
            packet = {}
            exec(compile(header, "finite packet", "exec"), packet)
        self.assertEqual(set(unpack(packet["PAYLOAD"]["files"])), set(package.source(package.REPO)[1]))
        queue = launch.Queue(fleet.repo / "chunk-fixture.sqlite")
        self.addCleanup(queue.db.close)
        with queue.db:
            claimed = launch.reserve_wave(queue, campaign, "a"*40, snapshot["local_receipt"],
                                          time.monotonic()+10, "f"*64, launch.DIRECTORY)
        self.assertEqual(set(claimed), set(campaign["shards"]))
        with self.assertRaises(sqlite3.IntegrityError), queue.db:
            launch.reserve_wave(queue, campaign, "a"*40, snapshot["local_receipt"],
                                time.monotonic()+10, "f"*64, launch.DIRECTORY)
        self.assertEqual(queue.summary()["runs"], {"DISPATCHING": len(claimed)})
        with self.assertRaisesRegex(ValueError, "new CPU-only dispatch snapshot"):
            package.select(fleet.verified_snapshot(age=91), set())


if __name__ == "__main__":
    unittest.main()
