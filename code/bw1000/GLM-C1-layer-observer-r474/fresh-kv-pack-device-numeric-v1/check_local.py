"""Check the component matrix, exact staging pins and admission without HIP."""

import ctypes as C
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from time import time
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P


class Buffer:
    def __init__(self, i, size):
        self.ptr = C.c_void_p(0x10000 * (i + 1))
        self.size = size
        self.damaged = False

    def upload(self, raw):
        self.payload = self.original = raw

    def readback(self):
        if self.damaged:
            raise RuntimeError("fake device guard changed")
        return self.payload

    def sentinel(self, stream):
        self.payload = b"\xff" * self.size

    def input_audit(self, tick):
        tick()
        if self.readback() != self.original:
            raise ValueError("fake input changed")


class Matrix:
    def __init__(self):
        self.buffers = [Buffer(i, size) for i, size in enumerate(P.SIZES)]
        self.by_address = {buf.ptr.value: buf for buf in self.buffers}
        self.stream = C.c_void_p(0x100000)
        self.syncs = 0
        self.called = 0
        self.rows = []

    def call(self, op, stream):
        assert op == "StreamSynchronize" and stream is self.stream
        self.syncs += 1

    def valid(self, key, value, projected, rotary, tokens, heads, stream):
        self.called += 1
        pointers = (key, value, projected, rotary)
        if not all(pointers) or stream is not self.stream or (tokens, heads) != (1, 4):
            return 1
        addresses = {pointer.value for pointer in pointers}
        if len(addresses) != 4 or not addresses <= self.by_address.keys():
            return 1
        k, v, p, r = (self.by_address[pointer.value] for pointer in pointers)
        expanded = struct.unpack("<1792H", p.payload)
        rotation = struct.unpack("<64H", r.payload)
        k.payload = struct.pack("<1024H", *(expanded[448 * h + c] if c < 192 else rotation[c - 192]
                                            for h in range(4) for c in range(256)))
        v.payload = struct.pack("<1024H", *(expanded[448 * h + c + 192]
                                            for h in range(4) for c in range(256)))
        return 0

    def exercise(self, exports=None):
        P.run_matrix(self, self.buffers, self.stream, exports or {tile: self.valid for tile in P.TILES},
                     lambda: None, lambda *row: self.rows.append(row))


class ProbeTests(unittest.TestCase):
    def test_frozen_references(self):
        repo = HERE.parents[3]
        for relative, digest in ((P.HELPER, P.HELPER_SHA), (P.KFD_HELPER, P.KFD_HELPER_SHA),
                                 (P.PLAN, P.PLAN_SHA), (P.ARCHIVE, P.ARCHIVE_SHA)):
            with self.subTest(path=relative):
                self.assertEqual(sha256((repo / relative).read_bytes()).hexdigest(), digest)

    def test_complete_matrix(self):
        matrix = Matrix()
        matrix.exercise()
        self.assertEqual((matrix.called, matrix.syncs), (40, 40))
        for group, denominator in P.PLANNED.items():
            passed = [row[1] for row in matrix.rows if row[0] == group and row[2] == "PASS"]
            self.assertEqual(passed, denominator)
        successes = [row[3] for row in matrix.rows if row[0] == "valid" and row[2] == "PASS"]
        self.assertEqual(successes[0], successes[1])
        self.assertEqual(successes[2], successes[3])
        self.assertNotEqual(successes[0], successes[2])

    def test_mutated_output_and_guard_preserve_failed_attempt(self):
        for mode in ("value_bit", "guard", "unchanged"):
            with self.subTest(mode=mode):
                matrix = Matrix()

                def bad(*args):
                    if mode == "unchanged":
                        return 0
                    result = matrix.valid(*args)
                    if result == 0:
                        buf = matrix.buffers[1]
                        if mode == "guard":
                            buf.damaged = True
                        else:
                            buf.payload = bytes([buf.payload[0] ^ 1]) + buf.payload[1:]
                    return result

                with self.assertRaises((RuntimeError, ValueError)):
                    matrix.exercise({64: bad, 128: matrix.valid})
                self.assertEqual(matrix.rows, [("valid", [17, 64], "STARTED")])

    def test_rejected_call_that_writes_is_detected(self):
        matrix = Matrix()

        def writes_on_reject(*args):
            result = matrix.valid(*args)
            if result == 1:
                matrix.buffers[0].payload = bytes(P.SIZES[0])
            return result

        with self.assertRaisesRegex(ValueError, "rejected argument changed"):
            matrix.exercise({64: writes_on_reject, 128: matrix.valid})
        self.assertEqual(matrix.rows[-1], ("rejected", [64, "null_key"], "STARTED"))

    def test_fresh_pre_admission(self):
        step = P.PARENT + ".999"
        now = 100_000_000_000
        scan = dict(kind="training_qwen_device_snapshot", mode="host_complete", node=P.NODE,
                    step=P.PARENT + ".998", pci=[f"0000:{90+i:02x}:00.0" for i in range(8)],
                    kfd_holder_count=0, full_fd_inventory_performed=True, boot_id="boot",
                    monotonic_ns=now - 1)
        self.assertEqual(P.verify_admission(scan, step, "boot", now), scan["pci"])
        for wrong in (dict(pci=scan["pci"][:-1]), dict(pci=[scan["pci"][0]] * 8),
                      dict(step=step), dict(monotonic_ns=now+1), dict(monotonic_ns=now-45_000_000_001),
                      dict(boot_id="other"), dict(kfd_holder_count=1), dict(kfd_holder_count=False),
                      dict(full_fd_inventory_performed=False)):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                P.verify_admission({**scan, **wrong}, step, "boot", now)

    def test_original_intent_and_provider_resolution(self):
        epoch = time() - 1
        scope = dict(label=P.LABEL, node=P.NODE, parent=P.PARENT, request=P.REQUEST, epoch=epoch,
                     deadline=epoch+P.WHOLE_SECONDS, work_deadline=epoch+P.WORK_SECONDS,
                     archive_SHA256=P.ARCHIVE_SHA, pre_admission_SHA256="a" * 64, replay_allowed=False)
        P.verify_intent(scope, P.NODE, P.PARENT + ".999", 4)
        for wrong in (dict(epoch=float("nan")), dict(replay_allowed=True),
                      dict(work_deadline=epoch+P.WHOLE_SECONDS), dict(pre_admission_SHA256="")):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                P.verify_intent({**scope, **wrong}, P.NODE, P.PARENT + ".999", 4)
        self.assertEqual(P.provider_paths("liba.so => /lib/liba.so (0xff)\n /lib/libb.so (0xff)"),
                         {"/lib/liba.so", "/lib/libb.so"})
        with self.assertRaises(ValueError):
            P.provider_paths("liba.so => not found")


if __name__ == "__main__":
    unittest.main()
