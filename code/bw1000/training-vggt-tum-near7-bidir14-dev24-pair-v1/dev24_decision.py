"""Decide whether a separately audited Dev24 result warrants another augmentation study."""
from math import isfinite


ARMS = ("base", "near7_bidir25")


def evaluate(independent):
    """Consume the unique follow-up's audited Dev24 independent evidence."""
    keys = ("development_AUC_at_5", "development_AUC_at_15", "development_AUC_at_30")
    if (independent.get("assessed_windows") != 24 or independent.get("input_pairs") != 22
            or independent.get("actual_prediction_events") != 44):
        return dict(decision="MISSING_OR_INCOMPLETE_DEV24", promotion=False)
    raw = independent.get("raw_pose_predictions", {})
    invalid = independent.get("invalid_predictions", {})
    if (set(raw) != set(invalid) or set(raw) != set(ARMS)
            or any(type(count) is not int or not 0 <= count <= 22 for count in (*raw.values(), *invalid.values()))
            or any(raw[arm] + invalid[arm] != 22 for arm in ARMS)):
        return dict(decision="MISSING_OR_INCOMPLETE_DEV24", promotion=False)
    summary = independent.get("AUC", {}).get("all_windows", {})
    for arm in ARMS:
        row = summary.get(arm, {})
        if (type(row.get("failed_pairs")) is not int or not 2 + invalid[arm] <= row["failed_pairs"] <= 24
                or any(type(row.get(key)) not in (float, int) or not isfinite(row[key])
                       or not 0 <= row[key] <= 1 for key in keys)):
            return dict(decision="MISSING_OR_INCOMPLETE_DEV24", promotion=False)

    baseline, candidate = [summary[arm] for arm in ARMS]
    passed = (candidate[keys[2]] >= baseline[keys[2]]
              and candidate[keys[1]] >= 0.95 * baseline[keys[1]]
              and candidate["failed_pairs"] <= baseline["failed_pairs"])
    return dict(decision="FOLLOWUP_MAY_BE_CONSIDERED" if passed else "STOP_NEAR_ONLY_AUGMENTATION",
                promotion=False, baseline=baseline, candidate=candidate)
