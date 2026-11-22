"""Exercise paired timing, guard failures and independent finish without a DCU."""

import ctypes as C
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import derive
import finish
import host_once as H
import result
import timing as T


REPO = HERE.parents[3]
PROBE, WAITER, TRES = H.verify_stage(REPO)
BOOT = "1e3ff586-fd62-4b74-bf36-0388e04c761c"
PCI = [f"0000:{183+i:02x}:00.0" for i in range(8)]
OWNER = {"pid": 4242, "startticks": 42917, "boot_id": BOOT}
MEM = dict(free_bytes=32*1024**3, total_bytes=64*1024**3)
HASHES = {tokens: T.fixed_hashes(tokens) for tokens in T.SHAPES}


def args_snapshot(step):
    return dict(kind="training_qwen_device_snapshot", mode="host_complete", node=H.NODE,
                step=step, pci=PCI, kfd_holder_count=0, full_fd_inventory_performed=True,
                boot_id=BOOT, monotonic_ns=10**9)


def trace_records(step, intent_sha, invalid=None):
    yield dict(state="STARTED", request=H.REQUEST, label=H.LABEL, step=step, node=H.NODE,
               intent_SHA256=intent_sha, planned=PROBE.PLANNED, native_SHA256=PROBE.NATIVE_SHA, **OWNER)
    yield dict(state="BOUND", providers={"count": 71, "canonical_HIP": str(PROBE.HIP_CANONICAL),
                                        "canonical_HIP_bytes": PROBE.HIP_BYTES}, hip_SHA256=PROBE.HIP_SHA,
               before=MEM, physical_device=dict(pci=PCI[0], visible_devices=1, kfd=dict(pci=PCI[0], gpu_id=10)))
    for identity in T.PLAN["arms"]:
        tok, pair, tile = identity
        yield dict(state="ARM_START", id=identity, planned_warm=T.WARM, planned_timed=T.TIMED[tok])
        if invalid == "gpu_failure" and identity == [1, 0, 128]:
            yield dict(state="ARM_FAIL", id=identity, type="ValueError", error="guard", warm_submitted=32, timed_submitted=5)
            yield dict(state="FAIL", type="ValueError", error="guard")
            return
        inputs, outputs = HASHES[tok]
        before = dict(MEM)
        if invalid == "different_memory" and identity == [1, 0, 128]:
            before["free_bytes"] += 4096
        yield dict(state="ARM_PASS", id=identity, input_SHA256=inputs, warm_output_SHA256=outputs,
                   timed_output_SHA256=outputs, guards="bitwise", before=before, after=MEM,
                   warm_submitted=T.WARM, timed_submitted=T.TIMED[tok], gpu_event_ms=10 if tile==64 else 8,
                   host_elapsed_ns=110, host_submit_ns=80, host_wait_ns=30, event_query_ns=10)
    yield dict(state="PASS", teardown="normal", during=MEM, after=MEM)


class Buffer:
    def __init__(self, payload):
        self.ptr = self
        self.payload = payload
        self.original = payload

    def sentinel(self, stream):
        self.payload = b"\xff" * len(self.payload)

    def input_audit(self, tick):
        tick()
        if self.original != self.payload:
            raise ValueError("input changed")

    def readback(self):
        return self.payload


class Timeline:
    def __init__(self):
        self.operations = []

    def mem(self):
        return MEM

    def call(self, op, *args):
        self.operations.append((op, *args))
        if op == "EventElapsedTime":
            C.cast(args[0], C.POINTER(C.c_float)).contents.value = 0.625


class PairTest(unittest.TestCase):
    def test_exact_hashes_shapes_and_derivation(self):
        self.assertIsNotNone(re.fullmatch("[0-9a-f]{32}", H.REQUEST))
        self.assertEqual(len(T.PLAN["arms"]), 12)
        for name, raw in derive.render().items():
            self.assertEqual((HERE / name).read_bytes(), raw)
        projected, rotary = T.fixture(512)
        key, value = T.reference(512, projected, rotary)
        self.assertEqual(tuple(map(len, (key, value, projected, rotary))), (1048576,1048576,1835008,65536))
        for token, head in ((0,3),(1,0),(510,2),(511,0),(511,3)):
            row = token*4+head
            self.assertEqual(key[row*512+384:(row+1)*512], rotary[token*128:(token+1)*128])
            self.assertEqual(value[row*512:(row+1)*512], projected[row*896+384:(row+1)*896])

    def test_event_interval_has_only_requested_launches(self):
        for tokens in T.SHAPES:
            with self.subTest(tokens=tokens):
                hip = Timeline()
                inputs = T.fixture(tokens)
                expected = T.reference(tokens,*inputs)
                buffers = [Buffer(item) for item in (*expected,*inputs)]
                logs = []

                def invoke(key, value, projected, rotary, count, heads, stream):
                    self.assertEqual((count, heads, stream), (tokens,4,"stream"))
                    key.payload, value.payload = expected
                    hip.operations.append(("actual_launch", count))
                    return 0

                T.run_arm(hip, buffers, ("stream","event0","event1"), invoke, [tokens,0,64], T.hashes(inputs), expected,
                          lambda: None, logs.append, clock=iter((20,40,50,70)).__next__)
                first = hip.operations.index(("EventRecord","event0","stream"))
                last = hip.operations.index(("EventRecord","event1","stream"))
                self.assertEqual(hip.operations[first+1:last], [("actual_launch",tokens)] * T.TIMED[tokens])
                self.assertEqual(logs[-1]["state"], "ARM_PASS")
                self.assertEqual(logs[-1]["gpu_event_ms"], 0.625)
                self.assertEqual((logs[-1]["host_submit_ns"],logs[-1]["host_wait_ns"],logs[-1]["event_query_ns"]), (20,10,20))

    def test_failed_warm_preserves_timed_denominator(self):
        hip = Timeline()
        inputs = T.fixture(1)
        expected = T.reference(1,*inputs)
        logs = []
        with self.assertRaisesRegex(ValueError, "full BF16"):
            T.run_arm(hip, [Buffer(item) for item in (*expected,*inputs)], ("s","e1","e2"),
                      lambda *_:0, [1,0,64], T.hashes(inputs), expected, lambda:None, logs.append)
        self.assertEqual(logs[0]["planned_timed"], 1024)
        self.assertEqual((logs[1]["state"],logs[1]["warm_submitted"],logs[1]["timed_submitted"]), ("ARM_FAIL",32,0))

    def scope(self, run, invalid=None):
        rows = [[H.PARENT+".980", "parent", "RUNNING", "0:0", "9000", H.NODE, "cpu=128,gres/dcu=8"]]

        def raw():
            return "".join("|".join(row)+"\n" for row in rows).encode()

        def runner(argv, *, stdout, **kwargs):
            phase = next(name for name in H.PHASES if "--job-name="+H.label(name) in argv)
            cards, cpus, _ = H.PHASES[phase]
            row = [H.PARENT+"."+str(980+len(rows)), H.label(phase), "COMPLETED", "0:0", "4", H.NODE,
                   f"cpu={cpus},mem=16G,node=1"+(",gres/dcu=1" if cards else "")]
            if phase == "gpu" and invalid == "gpu_failure":
                row[2:4] = ["FAILED","1:0"]
            rows.append(row)
            if phase == "gpu":
                intent = sha256((run/"intent.json").read_bytes()).hexdigest()
                (run/"probe.jsonl").write_bytes(b"".join(
                    (json.dumps(item)+"\n").encode() for item in trace_records(row[0],intent,invalid)))
            else:
                snap = args_snapshot(row[0])
                if phase == "post":
                    snap.update(worker_identity=json.loads(argv[argv.index("--worker")+1]), worker_state="OWNER_ABSENT")
                stdout.write((json.dumps(snap)+"\n").encode())
            return SimpleNamespace(returncode=int(row[2]!="COMPLETED"))

        clock = 1795300000.0
        def wait(read, deadline, **kwargs):
            return WAITER.wait_terminal(read, deadline, monotonic=lambda:17.0, **kwargs)

        outcome = H.launch(REPO, run, "a"*40, "b"*64, PROBE, SimpleNamespace(wait_terminal=wait), TRES,
                           query=lambda _:raw(), expires=lambda:clock+400, runner=runner,
                           wall=lambda:clock, mono=lambda:17.0,
                           mount=lambda *_: H.container(PurePosixPath("/incoming/paired"), PurePosixPath("/results/paired")))
        rows.append([H.PARENT+".984","independent-post","COMPLETED","0:0","1",H.NODE,"cpu=1,mem=256M,node=1"])
        payload = dict(request_id="5"*32, snapshot_kind="training80-dispatch-v1", complete=True, nodes={H.NODE:
                       dict(parent=H.PARENT, zone="O01", PCIs=PCI, active_steps=[], device_probe_observed=True,
                            membership_valid=True, KFD_clean=True, PID_clean=True, kfd_process_count=0,
                            own_device_fd_process_count=0, probe_error=None, device_probe_step=H.PARENT+".984")})
        payload["local_receipt"] = dict(request_id="5"*32, boot_epoch_anchor=clock-17, received_monotonic=16.,
                                        roundtrip_s=0.1, raw_response_SHA256="9"*64,
                                        payload_SHA256=finish.sha(finish.encoded(payload)))
        handle = dict(request_id=H.REQUEST, direct_kind="operator", intent="local-only/original.json",
                      placement_by_parent={H.PARENT:{H.NODE:1}})
        return outcome, raw(), payload, handle, clock

    def test_complete_and_failed_terminal_independent_finish(self):
        for invalid, expected in ((None,(12,0,0,True)),("gpu_failure",(1,1,10,False)),
                                  ("different_memory",(12,0,0,False))):
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                run = Path(directory)/"new"
                outcome, raw, snapshot, handle, clock = self.scope(run, invalid)
                self.assertTrue(outcome["cleanup_qualified"], outcome)
                observed = finish.audit(run, raw, snapshot, handle, PROBE, WAITER, TRES,T,result,
                                        now=lambda:17.,wall=lambda:clock)
                self.assertEqual(tuple(observed["finish"][key] for key in ("completed","failed","missing","passed")),expected)
                if invalid is None:
                    self.assertEqual([p["measured_pairs"] for p in observed["evidence"]["pairs_by_tokens"].values()],[3,3])
                if invalid=="different_memory":
                    self.assertEqual(observed["independent_trace"]["pairs_by_tokens"]["1"]["measured_pairs"],2)
                with self.assertRaisesRegex(ValueError, "stale"):
                    finish.audit(run, raw, snapshot, handle, PROBE, WAITER,TRES,T,result,now=lambda:117.,wall=lambda:clock+100)


if __name__ == "__main__":
    unittest.main()
