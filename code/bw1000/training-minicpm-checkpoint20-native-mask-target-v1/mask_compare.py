"""Compare four same-route forwards with and without the native mask observer."""

import math
import re

from mask_rows import compare as compare_masks
from route_control import consume_row, require


ARMS = ("u1", "c1", "c2", "u2")
CAP = 512 * 1024**2
MASKS = 186
OLDER = {"1103606.1171", "1103606.1189", "1103606.1194", "1103606.2303"}


def parse(arm, rank, step, record):
    owner = record["route_owner"]
    tap = record["native_mask_observation"]
    require(owner["arm"] == tap["arm"] == arm and owner["declared_policy"] == "NATIVE_AUTO"
            and tap["storage_cap"] == CAP, "observed arm, policy or mask cap differs")
    require(record["operators"]["native_dropout"] == {"aten::dropout": MASKS, "aten::native_dropout": MASKS},
            "original complete native dropout route differs")
    actual = tap["fingerprints"]
    if arm.startswith("u"):
        require(actual is None, "control must keep the original untapped forward")
    else:
        require(actual["native_mask_calls"] == len(actual["mask_fingerprints"]) == MASKS
                and 0 < actual["logical_mask_bytes_moved_to_host"] <= actual["charged_device_storage_bytes"]
                <= actual["cap_device_storage_bytes"] == CAP
                and actual["logical_mask_bytes_moved_to_host"] == sum(row["mask_bytes"] for row in actual["mask_fingerprints"])
                and type(actual["post_forward_copy_and_digest_seconds"]) is float
                and math.isfinite(actual["post_forward_copy_and_digest_seconds"])
                and actual["post_forward_copy_and_digest_seconds"] >= 0,
                "missing masks or invalid post-forward readback cost")
    # Both controls and capture use the original Flash policy. The existing
    # receipt consumer names that policy a1; validate the real tag first.
    parsed = consume_row("a1", rank, step, dict(record, route_owner=dict(owner, arm="a1")))
    parsed["held"]["autocast_and_dispatch"] = {
        "root": record["root_before_forward"]["context"],
        "language": record["language_before_forward"]["context"],
        "inner_linear": record["inner_linear"],
    }
    return dict(parsed, masks=actual)


def compare_rank(rank, arms, original):
    u1, c1, c2, u2 = (arms[arm] for arm in ARMS)
    held = u1["held"]
    require(all(row["held"] == held for row in (c1, c2, u2))
            and held["input"]["full_input"] == original["input"] and held["restored"] == original["state"]
            and held["initial_rng"] == original["RNG"] and held["attention_config"] == "sdpa",
            "input, restoration, RNG or forward context changed")
    mask = compare_masks(c1["masks"], c2["masks"])
    controls_exact = u1["loss"] == u2["loss"]
    captured_exact = c1["loss"] == c2["loss"]
    no_mask_difference = mask["different_native_masks"] == 0
    return dict(rank=rank, original_control_loss_exact=controls_exact, observed_capture_loss_exact=captured_exact,
                control_loss_gap=abs(u1["loss"][1] - u2["loss"][1]),
                capture_loss_gap=abs(c1["loss"][1] - c2["loss"][1]),
                control_end_rng_exact=u1["end_rng"] == u2["end_rng"],
                capture_end_rng_exact=c1["end_rng"] == c2["end_rng"],
                observer_and_control_end_rng_exact=all(row["end_rng"] == u1["end_rng"] for row in (c1, c2, u2)),
                mask_comparison=mask,
                repeated_loss_diff_with_equal_observed_masks=(not controls_exact and not captured_exact and no_mask_difference),
                control_variation_not_reproduced_with_observer=(not controls_exact and captured_exact),
                captured_device_storage_bytes=[row["masks"]["charged_device_storage_bytes"] for row in (c1, c2)],
                captured_post_forward_seconds=[row["masks"]["post_forward_copy_and_digest_seconds"] for row in (c1, c2)])


def comparison(records, step, history):
    require(re.fullmatch(r"1103606\.[0-9]+", step) and step not in OLDER, "require a new numbered mask diagnostic")
    ranks = set(range(8))
    require(set(history) == ranks and set(records) <= {(rank, arm) for rank in ranks for arm in ARMS},
            "original evidence or forward denominator differs")
    errors, rows, owners = [], [], []
    verified = 0
    for rank in range(8):
        parsed = {}
        for arm in ARMS:
            try:
                parsed[arm] = parse(arm, rank, step, records[rank, arm])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(dict(rank=rank, arm=arm, reason=type(error).__name__, detail=str(error)[:140]))
        verified += len(parsed)
        owners.extend(row["owner"] for row in parsed.values())
        if len(parsed) == len(ARMS):
            try:
                rows.append(compare_rank(rank, parsed, history[rank]))
            except (KeyError, TypeError, ValueError) as error:
                errors.append(dict(rank=rank, reason=type(error).__name__, detail=str(error)[:140]))
    if len(owners) != len(set(owners)) or len({owner[0] for owner in owners}) > 1:
        errors.append(dict(reason="RANK_PROCESS_REUSED_OR_BOOT_CHANGED"))
    return dict(actual_step=step, planned_forwards=32, verified_forwards=verified,
                missing_forwards=32 - len(records), comparable_ranks=len(rows), complete=not errors and len(rows) == 8,
                rows=rows, errors=errors, actual_optimizer_updates=0,
                causal_source_proven=False, formal_recovery_qualified=False)
