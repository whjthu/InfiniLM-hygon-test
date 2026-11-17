"""Inspect exact input-view names and call the frozen native entry's bound_inputs only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys

from scope import SRC, PREP, MEDIA, ARMS, INVENTORY_SHA, require, read, sha


def view(root, expected):
    actual, linked, truncated = set(), [], False
    for index, path in enumerate(root.rglob("*")):
        if index >= 256:
            truncated = True
            break
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            linked.append(relative)
        if path.is_file():
            actual.add(relative)
    def names(values):
        return [value if re.fullmatch(r"[A-Za-z0-9_./-]{1,160}", value) else "nonwhitelist-name-SHA256:" + hashlib.sha256(value.encode()).hexdigest()
                for value in sorted(values)]
    return dict(file_count=len(actual), expected_count=len(expected), missing=names(expected-actual),
                extra=names(actual-expected), linked=names(linked), traversal_truncated=truncated,
                member_names_SHA256=hashlib.sha256(("\n".join(sorted(actual))+"\n").encode()).hexdigest())


def probe(arm):
    parent, node = ARMS[arm]
    report = dict(arm=arm, expected_parent=parent, expected_node=node, passed=False,
                  expected_GPU_cards=0, actual_GPU_cards=None, generate_called=False, tokenize_called=False, explicit_SDK_calls=False,
                  default_CWD_cover_mechanism_proved=False)
    try:
        require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)"), "input gate must hold zero GPU")
        require(os.environ.get("SLURM_JOB_ID") == parent and socket.gethostname().split(".")[0] == node,
                "wrong CPU parent/node")
        report.update(actual_node=socket.gethostname().split(".")[0], actual_step=parent+"."+os.environ["SLURM_STEP_ID"])
        require(all(os.environ.get(key) in (None, "") for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")),
                "device visibility must be empty for the CPU entry gate")
        report["actual_GPU_cards"] = 0
        require("torch" not in sys.modules, "Torch was already imported; not a pure entry gate")
        frozen = read(SRC / "FROZEN_NATIVE.json")
        require(set(frozen) == {"native_generation.py", "check_explicit_dev_inputs.py", "check_qwen_native_inputs.py"},
                "frozen native entry source denominator differs")
        actual_sources = {name: sha(SRC / name) for name in frozen}
        require(actual_sources == frozen, "frozen native entry bytes differ")
        report["native_source_SHA256"] = actual_sources
        inventory = PREP / "inventory.json"
        require(sha(inventory) == INVENTORY_SHA, "new07inventory differs")
        pins = read(inventory)
        require(len(pins) == 39, "expected inventory is not39")
        given = PREP / "dev"
        resolved = given.resolve(strict=True)
        report.update(given_root=str(given), resolved_root=str(resolved), given_equals_resolved=given==resolved,
                      given=view(given, set(pins)), resolved=view(resolved, set(pins)))
        require(not report["resolved"]["traversal_truncated"], "unexpected broad input tree")
        # This is the actual imported native_generation.bound_inputs entry used by generate().
        sys.path.insert(0, str(SRC))
        module = runpy.run_path(str(SRC / "native_generation.py"), run_name="frozen_native_entry_gate")
        entry = module["bound_inputs"]
        require(Path(entry.__code__.co_filename) == SRC / "check_explicit_dev_inputs.py", "different native bound_inputs provider")
        visual, text = entry(resolved, inventory)
        require(len(visual) == len(text) == 108 and "torch" not in sys.modules, "native input gate imported Torch or lacks108rows")
        report.update(actual_native_entry="native_generation.bound_inputs", visual_rows=len(visual), text_only_rows=len(text),
                      actual_member_SHA256={name: sha(resolved / name) for name in sorted(pins)},
                      torch_imported=False, passed=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report.update(error_class=type(error).__name__, error=str(error)[:180], torch_imported="torch" in sys.modules)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.arm)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(arm=args.arm, passed=result["passed"], report=str(args.output),
                         actual_files=result.get("resolved", {}).get("file_count"), actual_GPU_cards=0)))
    raise SystemExit(0 if result["passed"] else 1)
