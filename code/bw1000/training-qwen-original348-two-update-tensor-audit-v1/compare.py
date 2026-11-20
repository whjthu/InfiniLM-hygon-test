"""Compare frozen per-rank fingerprints and scalar summaries from the original A100."""
import math


UPDATE_KEYS = (
    "preclip_gradient_l2", "postclip_gradient_l2", "reported_preclip_norm", "parameter_delta_l2",
)


def difference(got, expected):
    if not math.isfinite(got) or not math.isfinite(expected):
        raise ValueError("Nonfinite comparison value")
    return abs(got-expected), None if expected == 0 else abs(got-expected)/abs(expected)


def maximum(values):
    available = [v for v in values if v is not None]
    return max(available) if available else None


def scalar_comparison(target, reference):
    pairs = []
    updates = []
    if [x["rank"] for x in target] != list(range(8)) or [x["rank"] for x in reference["ranks"]] != list(range(8)):
        raise ValueError("Missing rank in comparison")
    for real, source in zip(target, reference["ranks"]):
        a, b = real["micro_forwards"], source["micro_forwards"]
        if len(a) != 8 or len(b) != 8 or len(real["updates"]) != 2 or len(source["updates"]) != 2:
            raise ValueError("Incomplete comparison denominator")
        for row, gold in zip(a, b):
            if ((row["global_step"], row["ordinal"], row["input_content"]["SHA256"]) !=
                    (gold["global_step"], gold["ordinal"], gold["input_content_SHA256"])):
                raise ValueError("Mismatched native record")
            loss_abs, loss_rel = difference(row["loss"], gold["loss"])
            _, output_l2_rel = difference(row["output_content"]["floating_l2"], gold["output_l2"])
            pairs.append(dict(rank=real["rank"], ordinal=row["ordinal"], loss_abs=loss_abs, loss_rel=loss_rel,
                              output_l2_rel=output_l2_rel,
                              exact_output_digest=row["output_content"]["SHA256"] == gold["output_content_SHA256"]))
        for got, gold in zip(real["updates"], source["updates"]):
            if got["update"] != gold["update"]:
                raise ValueError("Mismatched optimizer update")
            deltas = {k: difference(got[k], gold[k]) for k in UPDATE_KEYS}
            updates.append(dict(rank=real["rank"], update=got["update"], deltas=deltas,
                                learning_rates_equal=got["learning_rates"] == gold["learning_rates"],
                                changed_tensors_equal=got["changed_parameter_tensors"] == gold["changed_parameter_tensors"]))
    return dict(paired_micro_forwards=len(pairs), paired_rank_update_records=len(updates), exact_input_content=64,
                exact_output_digests=sum(p["exact_output_digest"] for p in pairs),
                max_loss_absolute=maximum([p["loss_abs"] for p in pairs]),
                max_loss_relative=maximum([p["loss_rel"] for p in pairs]),
                max_output_norm_relative=maximum([p["output_l2_rel"] for p in pairs]),
                max_update_scalar_absolute={k: maximum([p["deltas"][k][0] for p in updates]) for k in UPDATE_KEYS},
                max_update_scalar_relative={k: maximum([p["deltas"][k][1] for p in updates]) for k in UPDATE_KEYS},
                learning_rates_equal=sum(p["learning_rates_equal"] for p in updates),
                changed_tensor_counts_equal=sum(p["changed_tensors_equal"] for p in updates),
                a100_raw_tensor_elementwise=False, numerical_acceptance=False)
