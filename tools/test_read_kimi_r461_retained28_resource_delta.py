"""Exercise byte-level ELF resources, original score pins and unknown outcomes locally."""

import json
import struct
import unittest

import read_kimi_r461_retained28_resource_delta as audit


def message(value):
    if isinstance(value, dict):
        return bytes([0x80 + len(value)]) + b"".join(message(key) + message(item) for key, item in value.items())
    if isinstance(value, list):
        return bytes([0x90 + len(value)]) + b"".join(map(message, value))
    if isinstance(value, str):
        raw = value.encode()
        return (bytes([0xa0 + len(raw)]) if len(raw) <= 31 else bytes([0xd9, len(raw)])) + raw
    if type(value) is int:
        return bytes([value]) if value <= 0x7f else b"\xcd" + value.to_bytes(2, "big")
    raise TypeError(value)


def padded(raw):
    return raw + b"\x00" * ((-len(raw)) % 4)


def header(machine, *, table=0, count=0):
    magic = b"\x7fELF\x02\x01\x01".ljust(16, b"\0")
    return struct.pack("<16sHHIQQQIHHHHHH", magic, 1, machine, 1, 0, 0, table,
                       0, 64, 0, 0, 64, count, 0)


def image(kernel=None, *, scratch=0, vgpr=48, target="amdgcn-amd-amdhsa--gfx936", tag=32, duplicate=False):
    kernel = kernel or "_ZN12_GLOBAL__N_113scores_kernelE"
    descriptor = {".name": kernel, ".symbol": kernel + ".kd", ".vgpr_count": vgpr, ".sgpr_count": 32,
                  ".private_segment_fixed_size": scratch, ".group_segment_fixed_size": 1028,
                  ".wavefront_size": 64, ".max_flat_workgroup_size": 256}
    records = [descriptor, descriptor] if duplicate else [descriptor]
    document = {"amdhsa.kernels": records}
    if target is not None:
        document["amdhsa.target"] = target
    metadata = message(document)
    name = b"AMDGPU\0"
    note = struct.pack("<III", len(name), len(metadata), tag) + padded(name) + padded(metadata)
    sections = bytes(64) + struct.pack("<IIQQQQIIQQ", 0, 7, 0, 0, 64, len(note), 0, 0, 4, 0)
    return header(224, table=64 + len(note), count=2) + note + sections


def raw(document):
    return json.dumps(document, separators=(",", ":")).encode()


def acquisition(arm, image_bytes):
    request, label, node, manifest, total = audit.ARMS[arm]
    obj = header(62) + image_bytes
    step = "1103606." + ("450" if arm == "full-tree" else "451")
    row = [step, label, "COMPLETED", "0:0", "29", node, "cpu=4,mem=8G,node=1"]
    arm_probe = dict(object_SHA256=audit.digest(obj), fresh_M_SHA256="a" * 64, actual_MD_SHA256="b" * 64,
                     exports=["kimi_output_residual_scores_v1_launch"],
                     object=f"/refactor/candidates/{label}/{arm}/fused_scores.o")
    build = raw(dict(original_step=step, manifest_SHA256=manifest, node=node, planned_device_TUs=total,
                     actual_device_TUs=total, actual_GPU_steps=0, runtime_calls=0, passed=True, completed=True,
                     epoch=1795300000.5, deadline=1795300300.5,
                     all_fresh_M_before_compile=True, device_qualified=False, arms={arm: arm_probe}))
    terminal = raw(dict(columns=audit.COLUMNS, row=row, request=request, state="COMPILE_VERIFIED",
                        independent_observation=True, terminal_row_SHA256=audit.digest(raw(row)), observations=[row],
                        arms={arm: arm_probe}, score_receipt_SHA256=audit.digest(build)))
    return terminal, build, obj


class Retained28Resources(unittest.TestCase):
    def setUp(self):
        self.old = acquisition("full-tree", image())

    def test_exact_metadata_delta_keeps_device_and_performance_unknown(self):
        result = audit.compare(self.old, acquisition("retained28", image(vgpr=55, scratch=16)))
        self.assertTrue(result["actual_target_identity_verified"])
        self.assertEqual(result["resource_delta"]["vgpr_count"], 7)
        self.assertEqual(result["resource_delta"]["private_segment_fixed_size"], 16)
        self.assertFalse(result["no_added_fixed_scratch"])
        for field in ("observed_input_reload_eliminated", "spill_instruction_delta", "warmed_performance_passed"):
            self.assertIsNone(result[field])
        self.assertFalse(result["GPU_GO"])

    def test_fixed_scratch_same_does_not_prove_no_spill_instructions(self):
        result = audit.compare(self.old, acquisition("retained28", image(vgpr=63)))
        self.assertTrue(result["no_added_fixed_scratch"])
        self.assertIsNone(result["spill_instruction_delta"])

    def test_identical_target_images_exclude_an_instruction_improvement(self):
        result = audit.compare(self.old, acquisition("retained28", image()))
        self.assertTrue(result["identical_device_images"])
        self.assertFalse(result["observed_input_reload_eliminated"])
        self.assertFalse(result["GPU_GO"])

    def test_compressed_or_missing_metadata_reports_unknown(self):
        for payload in (b"CLANG_OFFLOAD_BUNDLE_COMPRESSED", image(tag=31), image(target=None)):
            with self.subTest(payload=payload[:16]):
                result = audit.compare(self.old, acquisition("retained28", payload))
                self.assertFalse(result["actual_target_identity_verified"])
                self.assertIsNone(result["resource_delta"])

    def test_wrong_arch_or_ambiguous_kernel_is_unresolved(self):
        for payload in (image(target="amdgcn-amd-amdhsa--gfx90a"), image(duplicate=True)):
            result = audit.compare(self.old, acquisition("retained28", payload))
            self.assertIn("metadata_unresolved", result["arms"]["retained28"])
            self.assertIsNone(result["resource_delta"])

    def test_changed_host_object_pin_rejects(self):
        terminal, receipt, obj = acquisition("retained28", image())
        with self.assertRaisesRegex(ValueError, "object or dependency pins"):
            audit.compare(self.old, (terminal, receipt, obj + b"different"))

    def test_other_parent_or_gres_or_unverified_controller_rejected(self):
        terminal, receipt, obj = acquisition("retained28", image())
        for key, replacement in ((0, "1087656.451"), (6, "cpu=4,mem=8G,node=1,gres/dcu=0,gres/gpu=1")):
            close = json.loads(terminal)
            close["row"][key] = replacement
            close["observations"] = [close["row"]]
            close["terminal_row_SHA256"] = audit.digest(raw(close["row"]))
            worker = json.loads(receipt)
            worker["original_step"] = close["row"][0]
            fixed_worker = raw(worker)
            close["score_receipt_SHA256"] = audit.digest(fixed_worker)
            with self.assertRaisesRegex(ValueError, "scope differs"):
                audit.compare(self.old, (raw(close), fixed_worker, obj))
        close = dict(json.loads(terminal), independent_observation=False)
        with self.assertRaisesRegex(ValueError, "scope differs"):
            audit.compare(self.old, (raw(close), receipt, obj))

    def test_bad_messagepack_rejects_duplicate_truncation_and_nesting(self):
        samples = (b"\x82\xa1x\x01\xa1x\x02", b"\x81\xa1x", b"\x81\xa1x" + b"\x91" * 20 + b"\x00")
        for payload in samples:
            with self.assertRaises(ValueError):
                audit.decode_metadata(payload)


if __name__ == "__main__":
    unittest.main()
