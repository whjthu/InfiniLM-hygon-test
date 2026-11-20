"""Bind the original .479 and .489 native head receipts without running a model."""

import argparse
import hashlib
import json
from pathlib import Path
import re


HEAD = "arms/continuous/train/camera_head.pt"
NODE = "f13r4n09"
BUNDLE = "b79a591250f5ba367438efc4122dcfe4a4ec66d59b821252e83b70404399c665"
ARMS = {
    "alternate25": ("1103606.479", "284947fbeb044a3f994f2627d04421cf", "1103606.485",
                    "759025dc9cd5e7ab64de3b4c89c67bd37983ddaa0e5edf67bfaff56fc6ea7cb7"),
    "lowlr25": ("1103606.489", "a0a20903a5f345d683cc1e8459b89800", "1103606.491",
                "e6159f1dad48df4f49e36363b3bb28e04bbac473afaeb98de21b71589cd2f72f"),
}
SOURCES = {
    "alternate25": ("training-vggt-tum-directed-translation-t12-alternate-v1",
                    "f53220165a356fec024505b65af2ac7f12b1c9272f6c231df22634c060f31ea5"),
    "lowlr25": ("training-vggt-tum-lowlr2e6-t12-v1",
                "888064324c11ce0e2df5ce11f27b7e26a9d8b400aa8cb96f0924059fd1256d32"),
}
UPDATES = {"continuous": 0, "resume10": 10, "resume20": 20}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def read_bounded(path, limit=2 * 1024**2):
    if path.resolve(strict=True) != path.absolute() or not path.is_file():
        raise ValueError("expected an original regular receipt")
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("unexpected original receipt size")
    return raw


def object_and_sha(path, wanted=None):
    raw = read_bounded(path)
    identity = digest(raw)
    if wanted is not None and identity != wanted:
        raise ValueError("original receipt SHA differs")
    return json.loads(raw), identity


def parse_accounting(raw):
    rows = {}
    for line in raw.decode("utf8").splitlines():
        row = [field.strip() for field in line.split("|")]
        if len(row) != 7:
            raise ValueError("expected seven explicit Slurm accounting columns")
        handle = row[0]
        if not re.fullmatch(r"1103606\.[0-9]+", handle):
            continue
        if handle in rows:
            raise ValueError("duplicate Slurm step identity")
        rows[handle] = row
    return rows


def verify_step(row, native, label=None):
    fields = [token.partition("=") for token in row[6].split(",")]
    tres = {key: val for key, eq, val in fields if key and eq and val}
    accelerators = {key: val for key, val in tres.items() if key.startswith("gres/")}
    if (not re.fullmatch(r"1103606\.[0-9]+", row[0]) or not row[1] or (label is not None and row[1] != label)
            or row[2:4] != ["COMPLETED", "0:0"] or not row[4].isdigit() or row[5] != NODE
            or len(tres) != len(fields) or tres.get("node") != "1"
            or accelerators != ({"gres/dcu": "1"} if native else {})):
        raise ValueError("original terminal Slurm row differs")


def bind_head(results, arm, accounting):
    gpu, request, clean, prior = ARMS[arm]
    label, backend = SOURCES[arm]
    root = results / label
    intent, intent_sha = object_and_sha(root / "intent.json")
    outcome, outcome_sha = object_and_sha(root / "outcome.json")
    if (intent["request"] != request or outcome["request"] != request or not intent["publication"]
            or outcome["publication"] != intent["publication"] or outcome["passed"] is not True
            or intent["parent"] != "1103606" or intent["node"] != NODE
            or intent["deadline"] - intent["epoch"] != 1800
            or not is_sha(outcome["result_SHA256"])):
        raise ValueError("original controller execution identity differs")

    steps = {}
    for phase in ("pre", "gpu", "post"):
        stored, _ = object_and_sha(root / (phase + "-accounting.json"))
        row = stored["row"]
        if (stored != outcome["stages"][phase] or stored["terminal_visible"] is not True
                or stored["passed"] is not True or stored["rc"] != 0 or accounting.get(row[0]) != row):
            raise ValueError("controller and Slurm terminal steps differ")
        verify_step(row, phase == "gpu", f"{label}-{phase}")
        steps[phase] = row[0]
    if steps["gpu"] != gpu or len(set(steps.values())) != 3 or clean not in accounting:
        raise ValueError("known training or independent cleanup step is missing")
    verify_step(accounting[clean], False)

    before, _ = object_and_sha(root / "before.json")
    after, _ = object_and_sha(root / "after.json")
    assigned, _ = object_and_sha(root / "assignment.json")
    if (len(before["pci"]) != 8 or len(set(before["pci"])) != 8 or before["pci"] != after["pci"]
            or before["boot_id"] != after["boot_id"] or assigned["step"] != gpu or assigned["node"] != NODE
            or assigned["pci"] not in before["pci"]
            or any(not proof["full_fd_inventory_performed"] or proof["kfd_holder_count"] != 0
                   or proof["step"] != steps[phase] or proof["node"] != NODE
                   for phase, proof in (("pre", before), ("post", after)))):
        raise ValueError("original placement or device cleanup differs")

    combined, combined_sha = object_and_sha(root / "native-result.json", outcome["result_SHA256"])
    if (arm == "alternate25" and combined_sha != prior or combined["actual_DCU"] != 1
            or combined["formal_CO3D_eligible"] is not False or combined["comparison"]["comparison_passed"] is not True
            or set(combined["receipts"]) != set(UPDATES)):
        raise ValueError("original native recovery evidence differs")
    continuous = None
    for phase, start in UPDATES.items():
        expected = combined["receipts"][phase]
        receipt, _ = object_and_sha(root / "arms" / phase / "receipt.json", expected["SHA256"])
        if (receipt != expected["value"] or receipt["arm"] != phase or receipt["backend_SHA256"] != backend
                or receipt["bundle_SHA256"] != BUNDLE or receipt["epoch"] != intent["epoch"]
                or receipt["deadline"] != intent["deadline"] - 100
                or receipt["executed_optimizer_steps"] != 25 - start
                or [step["global_optimizer_step"] for step in receipt["steps"]] != list(range(start + 1, 26))):
            raise ValueError("original training arm receipt identity differs")
        if start == 0:
            continuous = receipt
        elif receipt["final_state_SHA256"] != continuous["final_state_SHA256"]:
            raise ValueError("fresh-process recovery state differs")

    native, native_sha = object_and_sha(root / "arms/continuous/native-result.json", continuous["result_SHA256"])
    metrics = native["metrics"]
    if (metrics["global_optimizer_step"] != 25 or metrics["executed_optimizer_steps"] != 25
            or metrics["completed_global_step_before_resume"] != 0
            or metrics["checkpoint_identity"] != continuous["checkpoint_identity"]
            or metrics["final_state_sha256"] != continuous["final_state_SHA256"]
            or any(native["checks"].get(key) is not True for key in (
                "camera_head_weights_changed", "frozen_aggregator_unchanged", "main_forward_device",
                "nonzero_finite_camera_head_gradients", "resume_boundary"))):
        raise ValueError("expected the original completed continuous camera-head training")
    artifact, = [item for item in native["artifacts"] if item["path"] == "/pilot/" + HEAD]
    module_sha = metrics["camera_head_sha_after"]
    if not is_sha(artifact["sha256"]) or not is_sha(module_sha):
        raise ValueError("head file and module identity are required")
    head = root / HEAD
    if head.resolve(strict=True) != head.absolute() or not head.is_file() or head.stat().st_size <= 0:
        raise ValueError("original trained head file is unavailable")
    return dict(gpu_step=gpu, request=request, clean_post=clean, global_step=25, file_SHA256=artifact["sha256"],
                module_SHA256=module_sha, training_receipt_SHA256=prior, head_bytes_stat=head.stat().st_size,
                original_controller_post=steps["post"], original_intent_SHA256=intent_sha,
                original_outcome_SHA256=outcome_sha, original_combined_native_SHA256=combined_sha,
                original_continuous_result_SHA256=native_sha, actual_checkpoint_bytes_read=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "accounting", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    accounting_raw = read_bounded(args.accounting)
    accounting = parse_accounting(accounting_raw)
    result = {arm: bind_head(args.results, arm, accounting) for arm in ARMS}
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")
    with args.output.open("xb") as output:
        output.write(raw)
    print(json.dumps(dict(heads_bound=len(result), authority_SHA256=digest(raw),
                          accounting_SHA256=digest(accounting_raw), actual_checkpoint_bytes_read=0)))


if __name__ == "__main__":
    main()
