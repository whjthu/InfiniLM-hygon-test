"""Offline tests for the fixed-hour HOST ownership and interruption boundaries."""

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_training80_busy_long_host as host


PARENT = "1103606"
OTHER = "1194900"
STEP = PARENT + ".3030"
OTHER_STEP = OTHER + ".1200"
REQUEST = "1" * 32
LABEL = f"training80-long-busy-{REQUEST}-{PARENT}"
LABELS = {p: f"training80-long-busy-{REQUEST}-{p}" for p in (PARENT, OTHER)}
OWN = dict(PID=79, start_ticks=55)
NODES = {f"f12r2n{i:02d}" for i in range(1, 41)}
DEADLINE = 10**14


class HostTest(unittest.TestCase):
    def test_exact_live_parent_and_step_require_both_zero_gres_rows(self):
        child = SimpleNamespace(pid=OWN["PID"], poll=lambda: None)
        uid = 154459
        parent_line = f"JobId={PARENT} JobState=RUNNING NumNodes=40 UserId=test({uid}) AllocTRES=gres/dcu=320 NodeList=target"
        step_line = f"StepId={STEP} Name={LABEL} State=RUNNING TRES=cpu=40,mem=10G,node=40 NodeList=target"
        accounting = [STEP, LABEL, "RUNNING", "0:0", "1", "target", "cpu=40,mem=10G,node=40"]

        def evaluate(job=parent_line, step=step_line, row=accounting, ticks=55, handle=STEP, process=child):
            query = Mock(side_effect=[[job] if job else [], [step] if step else []])
            return host.qualified_cancel_step(PARENT, handle, LABEL, NODES, OWN, process, DEADLINE,
                query=query, account=lambda *_: row, hosts=lambda *args: NODES, ticks=lambda _: ticks, uid=lambda: uid)

        self.assertTrue(evaluate())
        self.assertFalse(evaluate(handle=None))
        self.assertFalse(evaluate(handle=PARENT))
        self.assertFalse(evaluate(handle=OTHER_STEP))
        self.assertFalse(evaluate(ticks=56))
        self.assertFalse(evaluate(process=None))
        self.assertFalse(evaluate(job=parent_line.replace("154459", "99999")))
        self.assertFalse(evaluate(step=step_line.replace(LABEL, "other-job")))
        self.assertFalse(evaluate(step=step_line.replace("State=RUNNING", "State=TIMEOUT")))
        self.assertFalse(evaluate(step=step_line.replace("node=40", "node=40,gres/dcu=8")))
        self.assertFalse(evaluate(row=None))
        self.assertFalse(evaluate(row=[*accounting[:6], "cpu=40,mem=10G,node=40,gres/dcu=8"]))

    def test_cancellation_invokes_only_exact_numeric_step_and_requires_gate(self):
        query = Mock()
        with patch.object(host, "qualified_cancel_step", wraps=host.qualified_cancel_step):
            self.assertEqual(host.cancel_owned(PARENT, STEP, LABEL, NODES, OWN, None,
                                               DEADLINE, gate=lambda *a: False, query=query), "UNVERIFIED_NO_SIGNAL")
        query.assert_not_called()
        self.assertEqual(host.cancel_owned(PARENT, STEP, LABEL, NODES, OWN, None, DEADLINE,
                                           gate=lambda *a: True, query=query), "EXACT_OWN_STEP_SIGNAL_SENT")
        query.assert_called_once_with(["scancel", STEP], DEADLINE, 65_536)

    def test_accounting_unknown_has_short_cleanup_bound(self):
        moment = [1_000_000_000]

        def pause(seconds):
            moment[0] += round(seconds * 1e9)

        known = [STEP, LABEL, "COMPLETED", "0:0", "1", "target", "cpu=40,mem=10G,node=40"]
        account = Mock(side_effect=lambda parent, *args: known if parent == PARENT else None)
        guard = Mock(return_value="UNVERIFIED_NO_SIGNAL")
        rows, actions, _ = host.finish({PARENT: STEP, OTHER: OTHER_STEP}, LABELS, host.window.PARENTS,
            {}, {}, DEADLINE, True, clock=lambda: moment[0], account=account, pause=pause, cancel_step=guard)
        self.assertEqual(rows, {PARENT: known})
        self.assertEqual(actions, {PARENT: "UNVERIFIED_NO_SIGNAL", OTHER: "UNVERIFIED_NO_SIGNAL"})
        self.assertLessEqual(moment[0] - 1_000_000_000, host.CLEANUP_LIMIT_NS)
        self.assertTrue(all(call.args[2] <= 1_000_000_000 + host.CLEANUP_LIMIT_NS for call in account.call_args_list))
        self.assertEqual([call.args[1] for call in guard.call_args_list], [STEP, OTHER_STEP])

    def test_absent_handshake_never_discovers_a_step_for_cancellation(self):
        guard = Mock(return_value="UNVERIFIED_NO_SIGNAL")
        acct = Mock()
        rows, actions, _ = host.finish({}, LABELS, host.window.PARENTS, {}, {}, DEADLINE, True,
            clock=lambda: 0, account=acct, cancel_step=guard)
        acct.assert_not_called()
        self.assertEqual(rows, {})
        self.assertTrue(all(state == "UNVERIFIED_NO_SIGNAL" for state in actions.values()))
        self.assertTrue(all(call.args[1] is None for call in guard.call_args_list))

    def test_verified_cancellation_precedes_pipe_shutdown_and_accounting(self):
        calls = Mock()
        calls.cancel.return_value = "EXACT_OWN_STEP_SIGNAL_SENT"
        calls.shutdown.return_value = []
        calls.account.return_value = [STEP, LABEL, "CANCELLED"]
        host.finish({PARENT: STEP}, LABELS, {PARENT: NODES}, {}, {}, DEADLINE, True,
                    clock=lambda: 0, account=calls.account, cancel_step=calls.cancel, shutdown=calls.shutdown)
        self.assertEqual([call[0] for call in calls.mock_calls], ["cancel", "shutdown", "account"])

    def test_release_requires_exact_end_and_no_controller_disagreement(self):
        terminal = "|".join([STEP, LABEL, "CANCELLED", "0:15", "1", "target",
                            "cpu=40,mem=10G,node=40", "2026-11-25T07:39:00"])
        active = f"StepId={STEP} Name={LABEL} State=RUNNING TRES=cpu=40,mem=10G,node=40 NodeList=target"
        def evaluate(record, controller):
            query = Mock(side_effect=[[record], controller])
            return host.released_step(PARENT, STEP, LABEL, NODES, DEADLINE, query=query, hosts=lambda *args: NODES)
        self.assertTrue(evaluate(terminal, []))
        self.assertFalse(evaluate(terminal, [active]))
        self.assertTrue(evaluate(terminal, [active.replace("State=RUNNING", "State=CANCELLED")]))
        self.assertFalse(evaluate(terminal.replace("2026-11-25T07:39:00", "Unknown"), []))
        self.assertFalse(evaluate(terminal.replace("node=40", "node=40,gres/dcu=1"), []))

    def test_stop_identity_and_nonblocking_writer_prevent_next_send(self):
        identity = dict(request=REQUEST, controller_PID=79, controller_start_ticks=55)
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            stopping = host.StopLatch(directory, identity)
            self.assertFalse(stopping())
            control = directory / "stop.request.json"
            control.write_text(json.dumps({**identity, "controller_start_ticks": 99,
                                           "action": "STOP_OBSERVER_FOR_PRIORITY_WORK"}))
            self.assertTrue(stopping())
            self.assertEqual(stopping.reason, "STOP_CONTROL_INVALID")
            control.write_text(json.dumps({**identity, "action": "STOP_OBSERVER_FOR_PRIORITY_WORK"}))
            stopping = host.StopLatch(directory, identity)
            self.assertTrue(stopping())
            self.assertEqual(stopping.reason, "PRIORITY_WORK")
        trans = object.__new__(host.PipeTransport)
        trans.clock = lambda: 1
        trans.children = {parent: SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: i))
                          for i, parent in enumerate(host.window.PARENTS, 3)}
        trans.buffers = dict.fromkeys(host.window.PARENTS)
        trans.stopping = Mock(side_effect=[False, True])
        with patch.object(host.os, "write", side_effect=lambda _, data: len(data)) as write:
            with self.assertRaises(TimeoutError):
                trans.send({"sequence": 1}, DEADLINE)
            self.assertEqual(write.call_count, 1)
        trans.stopping = lambda: False
        with patch.object(host.os, "write", return_value=1) as write:
            with self.assertRaisesRegex(ValueError, "partial"):
                trans.send({"sequence": 2}, DEADLINE)
            self.assertEqual(write.call_count, 1)
        trans.clock = lambda: DEADLINE
        with patch.object(host.os, "write") as write:
            with self.assertRaises(TimeoutError):
                trans.send({"sequence": 3}, DEADLINE)
            write.assert_not_called()

    def test_absolute_launch_window_and_frozen_worker_argv(self):
        as_ns = lambda label: int(datetime.fromisoformat(label).timestamp()) * host.NANO
        self.assertEqual(host.time_gate(lambda: 1000, lambda: as_ns(host.LAST)),
                         (1000, 1000, 1000 + as_ns(host.HARD_END) - as_ns(host.LAST)))
        with self.assertRaisesRegex(ValueError, "outside"):
            host.time_gate(lambda: 1000, lambda: as_ns(host.LAST) + 1)
        argv = host.command(PARENT, sorted(NODES), LABEL, REQUEST, Path("pci.json"))
        self.assertEqual(argv[0], "srun")
        self.assertIn("--gres=none", argv)
        self.assertIn("--time=01:01:00", argv)
        self.assertIn("--input=all", argv)
        self.assertIn(str(Path(host.__file__).resolve()), argv)
        self.assertEqual(argv[argv.index("--points") + 1], "1801")

    def test_first_stale_label_stops_before_starting_either_observer(self):
        # Use mock-only scheduler and clock. HOST namespace/evidence use the actual local filesystem.
        labels = {p: [] for p in host.window.PARENTS}
        scope = SimpleNamespace(name="posix", getpid=lambda: 79)
        receiver = SimpleNamespace(ready={}, parent_steps={})
        transport = Mock()
        transport.close.return_value = []
        def collection(*_args, **kwargs):
            self.assertTrue(kwargs["stopping"]())
            return {"hour_interval_coverage_complete": False}
        with (
            TemporaryDirectory() as temp, patch.object(host, "os", scope),
            patch.object(host, "pinned_paths", return_value=({}, {}, labels)),
            patch.object(host, "time_gate", return_value=(100, 120, DEADLINE)),
            patch.object(host, "proc_start_ticks", return_value=55),
            patch.object(host.short, "Receiver", return_value=receiver),
            patch.object(host.short, "check_live_parents"),
            patch.object(host.short, "find_accounting_step", return_value=STEP),
            patch.object(host.short, "qualified_accounting", return_value=False),
            patch.object(host, "PipeTransport", return_value=transport),
            patch.object(host.window, "collect_window", side_effect=collection),
            patch.object(host, "finish", return_value=({}, {}, [])),
            patch.object(host.subprocess, "Popen") as launch,
        ):
            folder = Path(temp) / ("training80-long-busy-" + REQUEST)
            fixture = Path(temp) / "fixture.json"
            fixture.write_text("{}")
            result = host.run(folder, fixture, fixture, REQUEST)
            launch.assert_not_called()
            self.assertEqual(result["startup_failure"], "ValueError")
            self.assertFalse(result["full_hour_observer_C0"])
            self.assertEqual(sorted(result["unresolved_steps"]), sorted(labels))
            self.assertTrue((folder / "host-outcome.json").exists())


if __name__ == "__main__":
    unittest.main()
