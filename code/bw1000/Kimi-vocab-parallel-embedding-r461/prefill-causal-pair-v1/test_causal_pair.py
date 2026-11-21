"""CPU typed-call fixtures for the paired continuation diagnostic."""

import copy
import ctypes as ct
import unittest

import numpy as np

from causal_pair import VOCAB, compare_logits, compare_run, digest, input_digest, run, schedule


class CapiFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


def vector(address, ctype, size):
    return np.ctypeslib.as_array(ct.cast(address, ct.POINTER(ctype)), shape=(size,))


class RecordingContext:
    def __init__(self, fault=None):
        self.fault = fault
        self.reset_count = 0
        self.history = []
        self.forward_count = 0
        self.actual_calls = []
        self.modern_kimi_engine_forward_ar_storage_r456 = CapiFunction(self.forward)
        self.modern_kimi_engine_reset_r2475 = CapiFunction(self.reset)
        self.blobs = {}
        self.events = []
        self.first_prefill = []

    def reset(self, _):
        self.reset_count += 1
        self.history.clear()
        return 7 if self.fault == "reset" else 0

    def forward(self, ctx, kind, width, batch, flags, *args):
        assert ctx == 908 and batch == 1 and flags == 3 and len(args) == 9
        assert vector(args[0], ct.c_int32, 1)[0] == 1
        past = int(vector(args[1], ct.c_int32, 1)[0])
        total = int(vector(args[2], ct.c_int32, 1)[0])
        offsets = vector(args[3], ct.c_int32, 2).tolist()
        ids = vector(args[4], ct.c_int64, width)
        positions = vector(args[5], ct.c_int64, width)
        assert offsets == [0, width] and past == len(self.history) and positions.tolist() == list(range(past, total))
        assert total == past + width and (kind == 0 or width == 1)
        self.history.extend(ids.tolist())
        self.actual_calls.append((self.reset_count, kind, past, ids.tolist()))
        self.forward_count += 1
        out = vector(args[6], ct.c_uint16, VOCAB)
        out[:] = np.uint16(0)
        selected = sum(self.history) % VOCAB
        out[selected] = np.uint16(0x3f80)
        if self.fault == "nonfinite":
            out[0] = np.uint16(0x7fc0)
        if self.fault == "mutation":
            positions[-1] += 1
        vector(args[7], ct.c_int64, 1)[0] = selected
        vector(args[8], ct.c_int32, 1)[0] = 3
        return 0

    def collect(self, label, packet):
        peers = [dict(packet, node=node) for node in (3, 1, 0, 2)]
        if ":call:" in label:
            for peer in peers:
                if peer["node"] != 0:
                    peer["selected"] = None
            if self.fault == "peer_position":
                peers[1]["positions_SHA256"] = "1" * 64
            if self.fault == "duplicate_peer":
                peers[1]["node"] = 3
        return peers

    def floor(self, _):
        return 4 << 30 if self.fault == "floor" else (4 << 30) + 1

    def sink(self, label, data):
        assert label not in self.blobs
        self.blobs[label] = data
        return dict(name=label, bytes=len(data), SHA256=digest(data))

    def inputs(self, count):
        return [i % 89 + 51 for i in range(count)], [777, 942]

    def exercise(self, count=1025, input_pin=None):
        prompt, suffix = self.inputs(count)
        pin = input_digest(prompt, suffix) if input_pin is None else input_pin
        return run(self, 908, 0, self, prompt, suffix, pin, "e" * 32, lambda: None,
                   self.floor, self.sink, lambda event, body: self.events.append((event, body)),
                   after_first_prefill=lambda: self.first_prefill.append(self.forward_count))

    def compare(self, result, count=1025):
        prompt, suffix = self.inputs(count)
        return compare_run(result, prompt, suffix, input_digest(prompt, suffix), lambda ref: self.blobs[ref["name"]])


class CausalPairTest(unittest.TestCase):
    def test_widths_share_full_prompt_and_continuation_at_four_positions(self):
        tap = RecordingContext()
        result = tap.exercise()
        observed = tap.compare(result)
        self.assertEqual(tap.reset_count, 4)
        self.assertEqual(tap.forward_count, 12)
        self.assertEqual(tap.first_prefill, [1])
        self.assertEqual(list(observed["compared"]), ["prefix512", "prompt-end", "continuation-1", "continuation-2"])
        self.assertEqual([p["context_tokens"] for p in observed["compared"].values()], [512, 1025, 1026, 1027])
        for row in observed["compared"].values():
            self.assertEqual(row["changed_BF16_bits"], 0)
            self.assertTrue(row["selected_token_matches"])
        prompt, suffix = tap.inputs(1025)
        for arm in (1, 3):
            self.assertEqual([i for call in tap.actual_calls if call[0] == arm for i in call[3]], prompt + suffix)
        self.assertFalse(observed["actual_KV_tensor_checked"])

    def test_short_prompt_and_full_context_schedule_preserve_budget(self):
        tap = RecordingContext()
        self.assertEqual(len(tap.compare(tap.exercise(512), 512)["compared"]), 3)
        for width in (256, 512):
            work = list(schedule(list(range(28672)), [777, 942], width))
            self.assertEqual(work[-1][1] + len(work[-1][2]), 28674)
            self.assertEqual(len(work), 28672 // width + 2)

    def test_invalid_receipts_are_rejected_and_failures_retained(self):
        for fault, pattern in (
            ("mutation", "modified host causal inputs"),
            ("nonfinite", "nonfinite native"),
            ("peer_position", "continuation metadata"),
            ("duplicate_peer", "TP32 node participation"),
            ("floor", "4 GiB floor"),
            ("reset", "initial reset failed"),
        ):
            with self.subTest(fault=fault):
                tap = RecordingContext(fault)
                with self.assertRaisesRegex(ValueError, pattern):
                    tap.exercise()
                self.assertEqual(tap.reset_count, 1 if fault == "reset" else 2)
                self.assertIn(("arm-error", dict(width=256, planned_calls=7, completed_calls=0)), tap.events)
                self.assertFalse(tap.blobs)
        tap = RecordingContext()
        with self.assertRaisesRegex(ValueError, "input identity differs"):
            tap.exercise(input_pin="a" * 64)
        self.assertEqual(tap.forward_count, 0)

    def test_reader_checks_private_bytes_and_schedule_before_comparison(self):
        tap = RecordingContext()
        result = tap.exercise()
        incomplete = copy.deepcopy(result)
        incomplete["arms"][0]["calls"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            tap.compare(incomplete)
        wrong = copy.deepcopy(result)
        wrong["arms"][0]["calls"][0]["total"] += 1
        with self.assertRaisesRegex(ValueError, "Causal call order"):
            tap.compare(wrong)
        name = next(iter(tap.blobs))
        tap.blobs[name] = bytes(len(tap.blobs[name]))
        with self.assertRaisesRegex(ValueError, "Bound logits changed"):
            tap.compare(result)

    def test_one_perturbed_bf16_value_is_detected(self):
        a = np.zeros(VOCAB, dtype="<u2")
        b = a.copy()
        b[21] = np.uint16(0x3f80)
        out = compare_logits(a.tobytes(), b.tobytes())
        self.assertEqual((out["changed_BF16_bits"], out["max_abs_delta"], out["same_argmax"]), (1, 1., False))


if __name__ == "__main__":
    unittest.main()
