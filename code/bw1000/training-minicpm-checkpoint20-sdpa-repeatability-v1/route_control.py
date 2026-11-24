"""Change only the native SDPA backend during one observed forward."""

from contextlib import contextmanager, nullcontext
import math
import re


ARMS = ("a1", "b1", "b2", "a2")
FLASH = "aten::_scaled_dot_product_flash_attention"
MATH = "aten::_scaled_dot_product_attention_math"
WRAPPER = "aten::scaled_dot_product_attention"
NATIVE_CALLS = 839


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mode(arm):
    require(arm in ARMS, "unexpected SDPA repeatability arm")
    return "MATH_ONLY" if arm.startswith("b") else "NATIVE_AUTO"


def flags(torch):
    return {name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("flash", "math", "mem_efficient", "cudnn")}


@contextmanager
def profile_with_route(torch, original_profiler, arm):
    selected = mode(arm)
    before = flags(torch)
    require(before["flash"], "native Flash route must be enabled before the comparison")
    context = (torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
               if selected == "MATH_ONLY" else nullcontext())
    try:
        with context:
            expected = ({"flash": False, "math": True, "mem_efficient": False, "cudnn": False}
                        if selected == "MATH_ONLY" else before)
            require(flags(torch) == expected, "SDPA policy is not bound to the chosen arm")
            with original_profiler() as profiler:
                yield profiler
    finally:
        require(flags(torch) == before, "SDPA flags survived the scoped forward")


def consume_row(arm, rank, step, record):
    require(record["rank"] == rank and record["world_size"] == 8
            and record["state"] == "STOP_BEFORE_BACKWARD"
            and record["checkpoint_step"] == record["global_step"] == 20
            and record["planned_forwards"] == record["completed_forwards"] == 1
            and record["backward_calls"] == record["optimizer_updates"] == 0,
            "not one native restored zero-update forward")
    owner = record["route_owner"]
    require(owner["arm"] == arm and owner["step"] == step and owner["declared_policy"] == mode(arm)
            and type(owner["pid"]) is int and owner["pid"] > 1
            and type(owner["startticks"]) is int and owner["startticks"] > 0 and owner["boot_id"],
            "native arm or exact process identity differs")
    require(record["actual_input"]["ordinal"] == record["actual_loss"]["ordinal"] == 1
            and record["actual_input"]["global_step"] == 20
            and record["actual_input"]["runtime_cursor"]["micro_in_update"] == 1,
            "wrong part of the original input")
    rng = record["RNG_after_native_restore"]
    require(rng == record["RNG_before_compute_loss"] == record["root_before_forward"]["RNG"],
            "RNG changed before the native model root")
    operators = record["operators"]
    expected = MATH if mode(arm) == "MATH_ONLY" else FLASH
    require(operators["SDPA_route_observed"] is True
            and operators["SDPA_wrappers"] == {WRAPPER: NATIVE_CALLS}
            and operators["selected_SDPA"] == {expected: NATIVE_CALLS},
            "actual selected SDPA route or call count differs")
    loss = record["actual_loss"]
    require(type(loss["loss_scalar"]) is float and math.isfinite(loss["loss_scalar"]), "invalid loss")
    return {"owner": (owner["boot_id"], owner["pid"], owner["startticks"]),
            "held": {"input": record["actual_input"], "restored": record["restored_state"], "initial_rng": rng,
                     "ddp": record["ddp"], "compute_loss": record["compute_loss_source"],
                     "language_training": record["language_before_forward"]["training"],
                     "attention_config": record["language_before_forward"]["attention_config"],
                     "root_module_modes": record["root_before_forward"]["model"]},
            "loss": (loss["loss"], loss["loss_scalar"]),
            "end_rng": record["RNG_after_forward"]}


def comparison(records, step, original_history):
    """Keep all 32 planned rank forwards in the denominator, including absent arms."""
    require(re.fullmatch(r"1103606\.[0-9]+", step) and step not in
            {"1103606.1171", "1103606.1189", "1103606.1194"}, "require a new actual numbered step")
    ranks = range(8)
    require(set(records) <= {(rank, arm) for rank in ranks for arm in ARMS}, "unexpected rank or arm")
    require(set(original_history) == set(ranks), "original rank evidence is incomplete")
    rows, errors, owners = [], [], []
    verified = 0
    for rank in ranks:
        parsed = {}
        for arm in ARMS:
            row = records.get((rank, arm))
            if row is None:
                errors.append({"rank": rank, "arm": arm, "error": "MISSING_RECORD"})
                continue
            try:
                parsed[arm] = consume_row(arm, rank, step, row)
            except (KeyError, TypeError, ValueError) as error:
                errors.append({"rank": rank, "arm": arm, "error": type(error).__name__, "reason": str(error)[:140]})
        verified += len(parsed)
        owners.extend(row["owner"] for row in parsed.values())
        if len(parsed) != len(ARMS):
            continue
        a1, b1, b2, a2 = (parsed[arm] for arm in ARMS)
        orig = original_history[rank]
        held = a1["held"]
        if (any(row["held"] != held for row in (b1, b2, a2)) or held["input"]["full_input"] != orig["input"]
                or held["restored"] != orig["state"] or held["initial_rng"] != orig["RNG"]
                or held["attention_config"] != "sdpa"):
            errors.append({"rank": rank, "error": "INPUT_RNG_STATE_OR_CONFIG_CHANGED"})
            continue
        rows.append(dict(rank=rank, flash_loss_exact=a1["loss"] == a2["loss"],
                         flash_end_rng_exact=a1["end_rng"] == a2["end_rng"],
                         math_loss_exact=b1["loss"] == b2["loss"], math_end_rng_exact=b1["end_rng"] == b2["end_rng"],
                         flash_loss_abs_gap=abs(a1["loss"][1]-a2["loss"][1]),
                         math_loss_abs_gap=abs(b1["loss"][1]-b2["loss"][1])))
    if len(owners) != len(set(owners)) or len({owner[0] for owner in owners}) > 1:
        errors.append({"error": "ARM_PROCESS_REUSE_OR_BOOT_CHANGED"})
    complete = not errors and len(rows) == 8
    return {"actual_step": step, "planned_forwards": 32, "verified_forwards": verified,
            "missing_forwards": 32 - len(records), "comparable_ranks": len(rows), "complete": complete,
            "rows": rows, "errors": errors, "actual_optimizer_updates": 0,
            "historical_cause_proven": False, "formal_recovery_qualified": False}
