"""Test the exact staged worker, dependency ordering, and bounded dispatch source."""

from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import derive
import launch
import package
from contract import NANO, TRANSFER_BYTES, counters, grid, validate_arm
from controller import unpack
from scope import EXCLUDED_NODES, MAX_NODES, PRIORITY_NODES, REQUEST
import test_report_training80_backlog as dispatch_fixture


class Device:
    def __init__(self):
        self.queues = {"compute": [], "copy": []}
        self.recorded = {}
        self.ready = set()
        self.calls = []
        self.graphs = 0
        self.copied = 0

    def drain(self, stream, through=None):
        queue = self.queues[stream]
        while queue:
            op, value = queue.pop(0)
            if op == "graph":
                self.graphs += 1
            elif op == "copy":
                self.copied += value
            elif op == "event":
                self.ready.add(value)
                if value == through:
                    return
            elif op == "wait":
                if value not in self.ready:
                    self.drain(self.recorded[value], through=value)
                assert value in self.ready

    def call(self, op, *args):
        self.calls.append((op, *args))
        if op == "EventRecord":
            marker, stream = args
            self.recorded[marker] = stream
            self.queues[stream].append(("event", marker))
        elif op == "StreamWaitEvent":
            stream, marker, flags = args
            assert flags == 0 and self.recorded[marker] == "compute"
            self.queues[stream].append(("wait", marker))
        elif op == "StreamSynchronize":
            self.drain(args[0])
        else:
            raise AssertionError(op)

    def push_graph(self):
        self.calls.append(("graph",))
        self.queues["compute"].append(("graph", 0))

    def push_copy(self, stream):
        self.calls.append(("D2H", stream, TRANSFER_BYTES))
        self.queues[stream].append(("copy", TRANSFER_BYTES))

    def query(self, _, window):
        self.calls.append(("query", window[0]))
        assert set(window[1:]) <= self.ready
        return 1.0


class EventJoinTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.target = Path(folder.name)
        _, self.files = package.source(package.REPO)
        for name, raw in self.files.items():
            (self.target / name).write_bytes(raw)
        self.worker = runpy.run_path(str(self.target / "worker.py"))

    def test_frozen_source_and_fixed_copy_from_original_measured_branch(self):
        for name, raw in derive.render(derive.ROOT).items():
            self.assertEqual((HERE / name).read_bytes(), raw)
        self.assertEqual(derive.sha(self.files["transfer.py"]), derive.ORIGINAL_TRANSFER_SHA)

    def test_real_unit_keeps_end_events_and_joins_both_device_queues(self):
        run = self.worker["timed_unit"]
        compute = ("compute", "compute_start", "compute_end")
        copy = ("copy", "copy_start", "copy_end")
        traces = []
        for mode in ("host_two_fences", "event_join", "common_host_two_fences"):
            device = Device()
            clock = SimpleNamespace(monotonic=iter(range(40)).__next__)
            transfer = SimpleNamespace(enqueue=device.push_copy)
            with patch.object(run.__globals__["prior"].dma, "elapsed_ms", side_effect=device.query):
                stats = run(device, compute, copy, (device.push_graph,)*2, transfer, mode, clock)
            self.assertEqual((device.graphs, device.copied, device.queues),
                             (2, TRANSFER_BYTES, {"compute": [], "copy": []}))
            self.assertEqual(stats["host_event_join_enqueue_ns"], int(mode == "event_join"))
            traces.append(device.calls)
        first, joined, common = traces
        self.assertEqual(first, common)
        self.assertEqual([x for x in first if x[0] == "StreamSynchronize"],
                         [("StreamSynchronize", "compute"), ("StreamSynchronize", "copy")])
        self.assertEqual([x for x in joined if x[0] in {"StreamWaitEvent", "StreamSynchronize"}],
                         [("StreamWaitEvent", "copy", "compute_end", 0), ("StreamSynchronize", "copy")])
        self.assertLess(joined.index(("EventRecord", "copy_end", "copy")),
                        joined.index(("StreamWaitEvent", "copy", "compute_end", 0)))
        untouched = lambda trace: [x for x in trace if x[0] not in {"StreamWaitEvent", "StreamSynchronize"}]
        self.assertEqual(untouched(first), untouched(joined))

    def test_wait_symbol_is_resolved_against_the_existing_pinned_library(self):
        bind = self.worker["bind_stream_wait_event"]
        scope = bind.__globals__
        fake_hip = SimpleNamespace(lib=object(), functions={})
        lookup = Mock()
        with (
            patch.object(scope["C"], "CDLL", return_value=SimpleNamespace(dladdr=lookup)),
            patch.dict(scope, symbol=Mock(return_value="bound")),
        ):
            bind(fake_hip)
            scope["symbol"].assert_called_once_with(fake_hip.lib, lookup, "hipStreamWaitEvent",
                                                    [scope["V"], scope["V"], scope["U"]], Path(scope["HIP_PATH"]))
        self.assertEqual(fake_hip.functions, {"StreamWaitEvent": "bound"})

    def test_event_join_counters_and_host_enqueue_subinterval(self):
        host, joined = counters("host_two_fences", 19), counters("event_join", 19)
        self.assertEqual((host.pop("stream_fences"), joined.pop("stream_fences")), (38, 19))
        self.assertEqual((host.pop("stream_event_waits"), joined.pop("stream_event_waits")), (0, 19))
        self.assertEqual(host, joined)
        windows = grid(dict(campaign_epoch=1000., shared_start_epoch=1082., deadline_epoch=1480.), 0)
        mode, start, end = windows[1]
        row = dict(index=1, mode=mode, scheduled_start_epoch_ns=start, scheduled_end_epoch_ns=end,
                   chains_per_submission=36, h2d_payload_bytes=0, state="COMPLETE", passed=True,
                   complete_groups_by_quarter=[5,5,5,4], actual_start_epoch_ns=start, actual_end_epoch_ns=end,
                   collection_end_epoch_ns=end, wall_ns=60*NANO, host_submit_ns=29, host_sync_wait_ns=30,
                   host_event_query_ns=1, host_event_join_enqueue_ns=23, terminal_polarity=1,
                   polarity_changes_between_replays=37, projection_full_BF16=True, projection_CPU_first16=True,
                   consumer_full_CPU=True, transfer_payload_guards_verified=True, compute_stream_ms=1.,
                   copy_stream_ms=0.25, **counters(mode,19))
        self.assertEqual(validate_arm(row, 1, windows[1]), "COMPLETE")
        with self.assertRaisesRegex(ValueError, "invalid host event-join enqueue time"):
            validate_arm({**row, "host_event_join_enqueue_ns": 30}, 1, windows[1])
        with self.assertRaisesRegex(ValueError, "dependent work counters"):
            validate_arm({**row, "stream_event_waits": 18}, 1, windows[1])

    def test_fresh_protected_fleet_and_real_local_queue_deduplication(self):
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
        plan, body, members = package.prepare("a"*40, 1000., snapshot, set())
        self.assertEqual(plan["request"], REQUEST)
        self.assertEqual((len(plan["selected"]), len(plan["shards"])), (MAX_NODES, 31))
        allowed = set(snapshot["nodes"]) - EXCLUDED_NODES - PRIORITY_NODES
        self.assertEqual(set(plan["selected"]), allowed)
        self.assertIn("f13r1n12", plan["selected"])
        self.assertNotIn("f13r4n09", plan["selected"])
        self.assertIn(package.JOIN_BASELINE.as_posix(), members)
        self.assertLessEqual(len(body), 131072)
        packet = {}
        exec(compile(body.partition(b"\n")[0], "finite payload header", "exec"), packet)
        self.assertEqual(unpack(packet["PAYLOAD"]["files"]), self.files)
        protected = {"f12r2n13", "f13r1n12"}
        smaller, _, _ = package.prepare("a"*40, 1000., snapshot, protected)
        self.assertEqual(set(smaller["selected"]), allowed - protected)
        queue = launch.Queue(fleet.repo / "event-join-fixture.sqlite")
        self.addCleanup(queue.db.close)
        with queue.db:
            handles = launch.reserve_wave(queue, plan, "a"*40, snapshot["local_receipt"],
                                          time.monotonic()+10, "f"*64, launch.DIRECTORY)
        self.assertEqual(len(handles), 31)
        with self.assertRaisesRegex(ValueError, "new CPU-only dispatch snapshot"):
            package.select(fleet.verified_snapshot(age=91), set())


if __name__ == "__main__":
    unittest.main()
