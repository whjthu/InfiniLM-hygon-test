import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from read_gemma_domain_terminal import (
    DATASET_SHA, FRAMEWORK, LABEL, NODE, REQUIRED_CHECKPOINT_FILES, read_terminal,
)
from test_compare_gemma_domain_recovery import fixture as rank_fixture


REQUEST = "a" * 32
PUBLICATION = "b" * 40


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture(root):
    manifest = json.dumps(dict(registered_case="T02.sft", base_framework_commit=FRAMEWORK,
        required_state=dict(world_size=8, continuous_updates=25, resume_from=20, resumed_updates=5))).encode()
    steps = dict(pre="1103606.810", gpu="1103606.811", post="1103606.812")
    binding = dict(request_id=REQUEST, publication=PUBLICATION, label=LABEL, steps=steps,
                   framework_base=FRAMEWORK, dataset_SHA256=DATASET_SHA, master_port=29920,
                   pair_manifest_SHA256=digest(manifest))
    write_json(root / "intent.json", binding)
    columns = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
    rows = {
        phase: [step, f"{LABEL}-{phase}", "COMPLETED", "0:0", "1", NODE,
                "cpu=64,node=1,gres/dcu=8" if phase == "gpu" else "cpu=2,node=1"]
        for phase, step in steps.items()
    }
    write_json(root / "terminal-accounting.json", dict(request_id=REQUEST, columns=columns, rows=rows))
    for phase, name, time in (("pre", "before.json", 100), ("post", "after.json", 900)):
        write_json(root / name, dict(node=NODE, step=steps[phase], pci=[f"0000:{i:02x}:00.0" for i in range(1, 9)],
            boot_id="11111111-1111-1111-1111-111111111111", kfd_holder_count=0, full_fd_inventory_performed=True,
            master_port=29920, master_port_available=True, monotonic_ns=time,
            owned_registry_present=True, owned_processes_remaining=[]))
    rank_fixture(root)
    for arm, checkpoints in (("continuous", [10, 20, 25]), ("resumed", [25])):
        kind = "continuous-witness" if arm == "continuous" else "resume-witness"
        ranks = []
        for rank in range(8):
            folder = root / arm / "train" / kind / f"rank-{rank:02d}"
            ranks.append(dict(rank=rank, result_SHA256=digest((folder / "result.json").read_bytes()),
                              owner_SHA256=digest((folder / "owner.json").read_bytes())))
        write_json(root / arm / "native-outcome.json",
                   dict(request_id=REQUEST, step=steps["gpu"], passed=True, rc=0, ranks=ranks,
                        owned_processes_remaining=0))
        for step in checkpoints:
            checkpoint = root / arm / "train" / f"checkpoint-{step}"
            checkpoint.mkdir()
            pins = {}
            for name in REQUIRED_CHECKPOINT_FILES:
                payload = (json.dumps(dict(global_step=step, max_steps=25, train_batch_size=1)).encode()
                           if name == "trainer_state.json" else b"simulated payload")
                (checkpoint / name).write_bytes(payload)
                pins[name] = dict(SHA256=digest(payload), bytes=len(payload))
            write_json(root / f"{arm}-checkpoint-{step}-files.json", dict(global_step=step, files=pins))
    return manifest, dict(request=REQUEST, publication=PUBLICATION, intent_sha=digest((root / "intent.json").read_bytes()))


class GemmaOriginalTerminal(unittest.TestCase):
    def test_saved_original_receipt_chain_and_independent_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, binding = fixture(root)
            result = read_terminal(root, manifest, **binding)
            self.assertTrue(result["passed"])
            self.assertEqual(result["steps"]["gpu"]["JobIDRaw"], "1103606.811")
            self.assertEqual(result["rank_comparison"]["compared_tail_microbatch_pairs"], 160)
            self.assertFalse(result["checkpoint_tensor_payloads_independently_hashed"])

    def test_rejects_live_or_wrong_allocation_and_dirty_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, binding = fixture(root)
            account = root / "terminal-accounting.json"
            original = account.read_bytes()
            value = json.loads(original)
            value["rows"]["gpu"][2] = "RUNNING"
            write_json(account, value)
            with self.assertRaisesRegex(ValueError, "no successful Slurm step"):
                read_terminal(root, manifest, **binding)
            value["rows"]["gpu"][2] = "COMPLETED"
            value["rows"]["gpu"][6] = "cpu=64,node=1,gres/dcu=4"
            write_json(account, value)
            with self.assertRaisesRegex(ValueError, "resource or label"):
                read_terminal(root, manifest, **binding)
            account.write_bytes(original)
            after = root / "after.json"
            value = json.loads(after.read_bytes())
            value["owned_processes_remaining"] = [123]
            write_json(after, value)
            with self.assertRaisesRegex(ValueError, "post-cleanup"):
                read_terminal(root, manifest, **binding)

    def test_rejects_unbound_rank_or_corrupted_trainer_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, binding = fixture(root)
            outcome = root / "continuous/native-outcome.json"
            original = outcome.read_bytes()
            value = json.loads(original)
            value["ranks"][4]["result_SHA256"] = "e" * 64
            write_json(outcome, value)
            with self.assertRaisesRegex(ValueError, "independently read rank receipt"):
                read_terminal(root, manifest, **binding)
            outcome.write_bytes(original)
            adapted = root / "continuous/train/checkpoint-20/trainer_state.json"
            write_json(adapted, dict(global_step=19, max_steps=25, train_batch_size=1))
            with self.assertRaisesRegex(ValueError, "actual Trainer state"):
                read_terminal(root, manifest, **binding)


if __name__ == "__main__":
    unittest.main()
