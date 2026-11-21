"""Independently compare the saved step-25 adapters from the USGS recovery pair."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from read_qwen417_adapter_drift import RESULTS, compare, need
from read_training_slurm_membership import allocated_dcu, completed_numbered_step


LABEL = "training-qwen-usgs-exposure25-resume20-hip-v1"
REQUEST = "f572fda6784140cbbdcae7bb91a916b9"
NODE = "f13r1n10"
CHECKPOINT = "adapter_model.safetensors"
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]
INVENTORY = {CHECKPOINT, "adapter_config.json", "args.json", "optimizer.pt", "scheduler.pt", "trainer_state.json"}
INVENTORY.update(f"rng_state_{rank}.pth" for rank in range(8))


def bounded(path, maximum):
    need(path.resolve(strict=True) == path and path.is_file(), "expected original canonical file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    need(0 < len(raw) <= maximum, "original evidence exceeds the byte limit")
    return raw


def gpu_step(accounting, controller):
    rows = [line.split("|") for line in accounting.decode("utf-8").splitlines() if line.strip()]
    candidates = [row for row in rows if len(row) == len(COLUMNS) and row[1] == LABEL + "-gpu"]
    need(len(candidates) == 1, "expected exactly one independently accounted GPU step")
    saved = controller["stages"]["gpu"]
    observed = candidates[0]
    need(saved["row"] == observed and saved["passed"] is True and saved["terminal_visible"] is True
         and saved["rc"] == 0, "controller and original accounting differ")
    info = completed_numbered_step(COLUMNS, observed, observed[0], scope="Qwen recovery")
    tres = dict(field.split("=", 1) for field in info["AllocTRES"].split(","))
    need(re.fullmatch(r"1103606\.\d+", info["JobIDRaw"]) and info["NodeList"] == NODE
         and info["ElapsedRaw"].isdecimal() and int(info["ElapsedRaw"]) <= 2100
         and tres.get("cpu") == "64" and tres.get("node") == "1"
         and tres.get("mem") in ("224000M", "218.75G") and allocated_dcu(info["AllocTRES"]) == 8,
         "original GPU placement or resource scope differs")
    return info["JobIDRaw"]


def adapter_pin(root, arm):
    inventory = json.loads(bounded(root / f"{arm}-checkpoint-25-files.json", 32 * 1024))
    files = inventory["files"]
    need(inventory["arm"] == arm and inventory["global_step"] == 25
         and inventory["member_count"] == len(INVENTORY) and files.keys() == INVENTORY,
         "saved checkpoint inventory differs")
    observed = files[CHECKPOINT]
    need(type(observed["bytes"]) is int and 8 < observed["bytes"] <= 512 * 1024**2
         and re.fullmatch("[0-9a-f]{64}", observed["SHA256"]), "original adapter file pin differs")
    return dict(bytes=observed["bytes"], sha256=observed["SHA256"])


def run(root, accounting, metadata_raw, metadata_sha):
    end = time.monotonic() + 180

    def check_time():
        if time.monotonic() >= end:
            raise TimeoutError("paired adapter comparison budget expired")

    need(sha256(metadata_raw).hexdigest() == metadata_sha, "independent metadata report identity differs")
    report = json.loads(metadata_raw)
    intent = json.loads(bounded(root / "intent.json", 64 * 1024))
    controller = json.loads(bounded(root / "controller-outcome.json", 512 * 1024))
    need(report["state"] == "QWEN_USGS25_AND_RESUME20_INDEPENDENT_METADATA_PASS" and report["passed"] is True
         and report["request"] == intent["request_id"] == REQUEST
         and report["publication"] == intent["publication"]
         and report["source_manifest_SHA256"] == intent["source_manifest_SHA256"]
         and report["actual_global_optimizer_invocations"] == 30 and report["actual_GPU_cards"] == 8
         and report["paired_microbatches"] == 160 and controller["passed"] is True,
         "independent recovery metadata or request identity differs")
    step = gpu_step(accounting, controller)
    pins = {side: adapter_pin(root, arm) for side, arm in (("stock", "continuous"), ("paired", "resumed"))}
    continuous = root / "continuous/train/checkpoint-25" / CHECKPOINT
    resumed = root / "resumed/train/checkpoint-25" / CHECKPOINT
    summary, _ = compare(continuous, resumed, pins, check_time)
    adapter = summary["all"]
    numeric_equal = adapter["differing_values"] == 0
    return dict(state="SAVED_ADAPTER_VALUES_EQUAL" if numeric_equal else "SAVED_ADAPTER_VALUE_DIFFERS",
                paired_numeric_equal=numeric_equal, request=REQUEST, original_GPU_step=step,
                terminal_accounting_SHA256=sha256(accounting).hexdigest(), metadata_report_SHA256=metadata_sha,
                publication=report["publication"], compared_checkpoints=2, compared_global_step=25,
                checkpoint_adapters=pins, adapter_stats=summary, independent_optimizer_or_rng_tensor_reads=0,
                new_model_invocations=0, new_GPU_steps=0, formal_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--terminal-metadata", type=Path, required=True)
    parser.add_argument("--terminal-metadata-sha256", required=True)
    args = parser.parse_args()
    result = run(RESULTS / LABEL, bounded(args.accounting, 1024 * 1024),
                 bounded(args.terminal_metadata, 128 * 1024), args.terminal_metadata_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return int(not result["paired_numeric_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
