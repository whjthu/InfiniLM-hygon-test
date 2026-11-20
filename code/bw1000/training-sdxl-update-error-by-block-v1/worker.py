"""Localize first-update disagreement to contiguous SDXL UNet module families."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re

import torch

from archived_step import checked_tensor, group, load, require, sha, squared

GATE = 0.05
BASELINE = 0.19271900631141356


def block(name):
    parts = re.findall(r"(?:^|\.)(down_blocks\.\d+|up_blocks\.\d+|mid_block)(?=\.|$)", name)
    require(len(parts) <= 1, "AMBIGUOUS_UNET_BLOCK")
    return parts[0] if parts else "other"


def metrics():
    return dict(tensors=0, elements=0, opposite_sign=0, one_zero=0, error_squared=0.0,
                opposite_sign_error_squared=0.0, reference_squared=0.0)


def add(destination, row):
    for key, value in destination.items():
        destination[key] = value + row[key]


def report(totals, groups):
    denominator = totals["reference_squared"]
    error = totals["error_squared"]
    require(denominator > 0 and math.isclose(math.sqrt(error / denominator), BASELINE, rel_tol=1e-9),
            "ORIGINAL_NATIVE_UPDATE_BOUNDARY")
    needed = error - GATE**2 * denominator
    ordered = sorted(groups, key=lambda name: (-groups[name]["error_squared"], name))
    cumulative = 0.0
    removal = []
    for name in ordered:
        item = groups[name]
        cumulative += item["error_squared"]
        removal.append(dict(scope=name, squared_error_share=item["error_squared"] / error,
                            ideal_remaining_relative_l2=math.sqrt(max(0.0, error - cumulative) / denominator)))
    return dict(required_error_squared_share=needed / error, ordered_ideal_removal=removal,
                minimum_scopes_if_all_errors_in_selected_scopes_disappeared=next(
                    (i for i, row in enumerate(removal, 1) if row["ideal_remaining_relative_l2"] <= GATE), None))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(torch.version.hip and torch.cuda.device_count() == 0 and not torch.cuda.is_initialized(),
            "CPU_ONLY_ORIGINAL_HIP_IMAGE")
    claim = json.loads(Path("/work/CLAIM.json").read_bytes())
    require(sha(Path("/reference/policy.json")) == claim["policy_sha256"], "ORIGINAL_POLICY")
    ref = load(Path("/reference/reference.pt"), claim["archives"]["A100"])
    observed = load(Path("/observed/observed.pt"), claim["archives"]["Hygon"])
    names = list(ref["initial_adapter"])
    keys = ("initial_adapter", "final_adapter", "gradients_before_clip", "gradients_after_clip")
    require(len(names) == 1120 and all(list(book[key]) == names for book in (ref, observed) for key in keys),
            "EXACT_ALL_1120_TRAINABLES")
    require(group(ref, len(names)) == group(observed, len(names)), "IDENTICAL_ADAMW_OPTIONS")
    totals = metrics()
    blocks = defaultdict(metrics)
    by_module = defaultdict(metrics)
    for name in names:
        initial = ref["initial_adapter"][name]
        a, b = [checked_tensor(book["initial_adapter"], name, initial.shape) for book in (ref, observed)]
        require(torch.equal(a, b), "EXACT_INITIAL_VALUES")
        preclip = [checked_tensor(book["gradients_before_clip"], name, a.shape) for book in (ref, observed)]
        g0, g1 = [checked_tensor(book["gradients_after_clip"], name, a.shape) for book in (ref, observed)]
        require(torch.equal(g0, preclip[0]) and torch.equal(g1, preclip[1]), "ORIGINAL_CLIP_IDENTITY")
        final0, final1 = [checked_tensor(book["final_adapter"], name, a.shape) for book in (ref, observed)]
        reference_update = final0.double() - a.double()
        difference = final1.double() - final0.double()
        opposite = g0.sign() * g1.sign() < 0
        row = dict(tensors=1, elements=a.numel(), opposite_sign=int(opposite.count_nonzero()),
                   one_zero=int(((g0 == 0) ^ (g1 == 0)).count_nonzero()),
                   error_squared=squared(difference), opposite_sign_error_squared=squared(difference[opposite]),
                   reference_squared=squared(reference_update))
        add(totals, row)
        add(blocks[block(name)], row)
        key = name.split(".lora_", 1)[0]
        require(key != name, "ORIGINAL_LORA_PARAMETER_NAME")
        add(by_module[key], row)
    require(totals["elements"] == 5806080 and totals["opposite_sign"] == 27690 and not totals["one_zero"],
            "EXACT_ARCHIVED_DENOMINATOR")
    require(all(math.isclose(sum(group[key] for group in blocks.values()), value, rel_tol=1e-12)
                for key, value in totals.items()), "BLOCK_PARTITION")
    block_removal = report(totals, blocks)
    module_removal = report(totals, by_module)
    kept = module_removal.pop("ordered_ideal_removal")[:12]
    module_removal["largest_12_modules"] = kept
    result = dict(case=claim["case"], request=claim["request"], totals=totals, blocks=blocks,
                  idealized_block_counterfactual=block_removal, idealized_module_counterfactual=module_removal,
                  actual_step=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                  GPU_initialized=torch.cuda.is_initialized(), physical_GPU=0,
                  real_update=False, error_gate=GATE)
    with Path("/work/result.json").open("x", encoding="utf8") as output:
        json.dump(result, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
