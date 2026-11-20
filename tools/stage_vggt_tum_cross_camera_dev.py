"""Stage the fixed VGGT development backfill on the Slurm login host, without submitting a step."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from bind_vggt_tum_cross_camera_heads import ARMS, SOURCES, bind_head, digest, parse_accounting
from bundle_vggt_tum_cross_camera_dev import MANIFEST, unpack


BUNDLE_SHA = "5b8f0340734a3c523e97a6b7adb17b46be0d68501fb0ea7c437130f0a9988460"
BUNDLE_BYTES = 5077419
INPUTS_SHA = "3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6"
COHORT = (22, 24, 23)
STAGE_NAME = "training-vggt-tum-cross-camera-dev24-v1-staged"
ACCOUNTING_ARGS = (
    "sacct", "-n", "-P", "-j", "1103606", "--starttime=2026-11-19T00:00:00",
    "--endtime=2026-11-22T00:00:00", "--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES",
)


def write_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf8")
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return digest(raw)


def stage(results, bundle, accounting_raw):
    output = results / STAGE_NAME
    if output.exists() or output.is_symlink():
        raise FileExistsError("the original stage namespace already exists; reconcile its completion receipt")
    if not 0 < len(accounting_raw) <= 2 * 1024**2:
        raise ValueError("expected bounded Slurm accounting for the original training")
    if bundle.stat().st_size != BUNDLE_BYTES:
        raise ValueError("expected the original private 24-window bundle size")
    accounting = parse_accounting(accounting_raw)
    heads = {arm: bind_head(results, arm, accounting) for arm in ARMS}

    weights = results / "training-vggt-original-public-model-v1/model/model.pt"
    if weights.resolve(strict=True) != weights.absolute() or not weights.is_file() or weights.stat().st_size != 5026874952:
        raise ValueError("expected the original VGGT base model file and byte count")
    upstream = results / "training-vggt-native-provider-HIP-CPU-v1/upstream"
    if upstream.resolve(strict=True) != upstream.absolute() or not (upstream / "vggt/models/vggt.py").is_file():
        raise ValueError("expected the original VGGT provider")

    output.mkdir(mode=0o700)
    inputs = output / "inputs"
    staged = unpack(bundle, BUNDLE_SHA, INPUTS_SHA, inputs)
    if tuple(staged[key] for key in ("input_pairs", "planned_windows", "verified_members")) != COHORT:
        raise ValueError("the staged development denominator differs")
    authority = output / "head-authority.json"
    authority_sha = write_json(authority, heads)
    with (output / "original-training-accounting.psv").open("xb") as receipt:
        receipt.write(accounting_raw)
    result = dict(
        state="INPUTS_STAGED_ORIGINAL_TRAINING_RECEIPTS_MATCHED", accounting_SHA256=digest(accounting_raw),
        head_authority_SHA256=authority_sha, input_staging=staged, actual_checkpoint_bytes_read=0,
        new_slurm_steps=0, optimizer_updates=0, GPU_device_admitted=False, scorer_permission=False,
        scorer_arguments=["--inputs", str(inputs / MANIFEST), "--inputs-sha", INPUTS_SHA,
                          "--head-authority", str(authority), "--head-authority-sha", authority_sha,
                          "--alternate", str(results / SOURCES["alternate25"][0]),
                          "--lowlr", str(results / SOURCES["lowlr25"][0]), "--upstream", str(upstream),
                          "--weights", str(weights), "--output", str(output / "score")],
    )
    marker = output / "stage-complete.json"
    pending = marker.with_suffix(".pending")
    marker_sha = write_json(pending, result)
    pending.replace(marker)
    return dict(state=result["state"], stage_receipt_SHA256=marker_sha, inputs=staged,
                original_heads_bound=len(heads), original_accounting_SHA256=result["accounting_SHA256"],
                actual_checkpoint_bytes_read=0, GPU_device_admitted=False, new_slurm_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.absolute()
    if results.resolve(strict=True) != results or not results.is_dir():
        parser.error("expected a canonical target results directory")
    accounting = subprocess.run(ACCOUNTING_ARGS, check=True, capture_output=True, timeout=20).stdout
    print(json.dumps(stage(results, args.bundle.absolute(), accounting), allow_nan=False))


if __name__ == "__main__":
    main()
