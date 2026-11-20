"""Check the formal same-input first-update gate against the original policy.

The matrix pins a case-specific comparison plan, which locates limits in the
original policy. The two execution receipts and the paired comparison are
separately pinned. Input digests represent the full realized microbatch,
including labels, masks, and any frozen conditioning or noise.
"""
import math

from training_formal_metrics import need, sha
from training_recovery import rank_values


BASE_CATEGORIES = {
    "prediction", "loss", "gradient_before_clip", "gradient_after_clip",
    "parameter_update", "optimizer_state",
}
POLICY_CATEGORIES = {"reward", "old_log_probability", "new_log_probability", "ratio", "advantage"}
POLICY_CASES = {"T06.ddpo", "T07.flow-grpo", "T08.motion-rft", "T09.sage-grpo"}
ZERO_LR_VLM_REQUESTS = {
    "T03.sft": "031e04f24efa429595364e6f0da1e5b6",
    "T04.sft": "640a2e0f2ae4440a9689912f2de413be",
}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def original_limit(policy, path):
    need(isinstance(path, list) and bool(path), "missing original threshold location")
    for part in path:
        need(isinstance(part, str) and isinstance(policy, dict) and part in policy,
             "threshold is absent from the pinned original policy")
        policy = policy[part]
    need(finite_number(policy), "original threshold is nonnumeric or nonfinite")
    return policy


def original_vlm_update_schedule(case, verify_raw):
    attachment = case.get("capture_value_evidence")
    need(isinstance(attachment, dict), "missing original A100 two-update VLM audit")
    reference = verify_raw(attachment)
    need(reference["original_training_request"] == ZERO_LR_VLM_REQUESTS[case["case"]]
         and reference["state"] == "ORIGINAL_CPU_VALUE_AUDIT_INDEPENDENT_TERMINAL_PASS"
         and reference["rank_denominator"] == reference["completed_ranks"] == 8
         and reference["optimizer_updates"] == 16 and reference["micro_forwards"] == 64
         and reference["full_artifact_SHA_and_finite"] is True
         and reference["initial_parameter_and_optimizer_boundary_continuity"] is True,
         "original A100 two-update VLM audit is missing or belongs to a different run")
    schedule = []
    for row in rank_values(reference["ranks"], 8):
        first, second = row["updates"]
        rates = second["optimizer_before"]["learning_rates"]
        need(first["update"] == 1 and second["update"] == 2
             and first["optimizer_before"]["learning_rates"] == [0.0]
             and first["parameters_changed"] == 0
             and first["parameters_denominator"] == second["parameters_denominator"] == 504
             and second["parameters_changed"] == 504
             and len(rates) == 1 and finite_number(rates[0]) and rates[0] > 0
             and first["optimizer_after"]["step"] == second["optimizer_before"]["step"] == 1
             and second["optimizer_after"]["step"] == 2
             and first["optimizer_after"]["state_values_SHA256"] == second["optimizer_before"]["state_values_SHA256"],
             f"A100 rank {row['rank']}: invalid zero-LR to first parameter update sequence")
        schedule.append(rates)
    return schedule


def comparison_plan(case, verify_raw):
    attachment = case.get("numerics_plan")
    need(isinstance(attachment, dict), "numerics requires the registered case-specific comparison plan")
    plan = verify_raw(attachment)
    case_id = case["case"]
    training_ranks = 1 if case_id == "T09.sage-grpo" else case["original_cards"]
    reference_commit = plan["A100_framework_commit"]
    need(plan["case"] == case_id and plan["original_launch_SHA256"] == case["launch_SHA256"]
         and sha(plan["original_launch_SHA256"])
         and isinstance(reference_commit, str) and len(reference_commit) == 40
         and all(c in "0123456789abcdef" for c in reference_commit)
         and type(plan["microbatches_per_rank"]) is int and plan["microbatches_per_rank"] > 0
         and type(plan["training_ranks"]) is int and plan["training_ranks"] == training_ranks,
         "numerics plan differs from the registered original launch or training geometry")
    policy = verify_raw(plan["original_policy"])
    reference = verify_raw(plan["original_reference"])
    required = {"before_update": {"prediction"}, "after_update": BASE_CATEGORIES}
    if case_id in POLICY_CASES:
        required["after_update"] = BASE_CATEGORIES | POLICY_CATEGORIES
    if case_id in ZERO_LR_VLM_REQUESTS:
        need(plan["training_ranks"] == 8 and plan["microbatches_per_rank"] == 4,
             "expected the original eight-rank VLM accumulation geometry")
        required["after_first_parameter_update"] = BASE_CATEGORIES
    need(set(plan["metrics"]) == set(required), "missing a required numerical comparison phase")
    if case_id in ZERO_LR_VLM_REQUESTS:
        need(plan["metrics"]["after_first_parameter_update"] == plan["metrics"]["after_update"],
             "VLM optimizer updates must use the same original numerical rules")
    rules = {}
    used_paths = set()
    for phase, categories in required.items():
        rows = plan["metrics"][phase]
        need(isinstance(rows, dict) and rows.keys()
             and all(isinstance(row, dict) for row in rows.values())
             and categories <= {row.get("category") for row in rows.values()},
             f"{phase}: required original metric categories are incomplete")
        rules[phase] = {}
        for name, row in rows.items():
            need(isinstance(name, str) and name and row.get("direction") in ("max", "min"),
                 f"{phase}: invalid metric name or threshold direction")
            path = row["policy_path"]
            limit = original_limit(policy, path)
            for direction in ("max", "min"):
                if path[-1].endswith("_" + direction):
                    need(row["direction"] == direction, f"{phase}: original threshold direction changed")
            rules[phase][name] = row["direction"], limit
            if phase == "after_update":
                used_paths.add(tuple(path))
    if case_id in POLICY_CASES:
        components = plan["reward_components"]
        reward_rules = [row.get("reward_component") for row in plan["metrics"]["after_update"].values()
                        if row["category"] == "reward"]
        need(isinstance(components, list) and components
             and all(isinstance(name, str) and name for name in components)
             and len(set(components)) == len(components)
             and set(reward_rules) == set(components),
             "first-update plan omits original reward components")
    if policy.get("schema") == "baseline-equivalence-policy-v1":
        need(reference["case"] == case_id and reference["framework_commit"] == reference_commit
             and reference["reference_SHA256"] == policy["reference_payload_sha256"]
             and reference["policy_SHA256"] == plan["original_policy"]["SHA256"],
             "original A100 replay does not bind the registered source, payload and numerical policy")
        required_paths = {(section, key)
                          for section in ("metrics", "posterior_additional_metrics")
                          for key, value in policy.get(section, {}).items()
                          if finite_number(value)}
        need(required_paths <= used_paths, "first-update plan omits original numerical thresholds")
        moments = {row.get("optimizer_state_component") for row in plan["metrics"]["after_update"].values()
                   if row["category"] == "optimizer_state"}
        need(set(policy.get("optimizer_state_tensor_keys", ())) <= moments,
             "first-update plan omits original optimizer moment tensors")
    return plan, policy, rules


def receipt(attachment, identity, shape, expected_cards, platform, verify_raw, update_schedule=None):
    value = verify_raw(attachment)
    need(all(value.get(key) == expected for key, expected in identity.items())
         and value["platform"] == platform and sha(value["run_id"]) and sha(value["terminal_receipt_SHA256"])
         and value["terminal_status"] == "COMPLETED"
         and type(value["cards"]) is int and value["cards"] == expected_cards
         and type(value["exit_code"]) is int and value["exit_code"] == 0,
         f"{platform}: wrong launch, source or terminal evidence")
    if identity["case"] == "T09.sage-grpo":
        roles = value["device_roles"]
        need(set(roles) == {"actor", "vae"} and all(type(device) is int for device in roles.values())
             and set(roles.values()) == {0, 1},
             f"{platform}: expected distinct actor and VAE devices")
    steps = value["optimizer_update_counts"]
    need(type(steps["zero_forward"]) is int and steps["zero_forward"] == 0
         and type(steps["first_update"]) is int and steps["first_update"] == 1,
         f"{platform}: expected a zero-update forward and one actual optimizer update")
    if update_schedule is not None:
        need(type(steps.get("first_parameter_update")) is int and steps["first_parameter_update"] == 2,
             f"{platform}: first parameter update must be optimizer step 2")
    ranks, micros = shape
    for row in rank_values(value["ranks"], ranks):
        need(sha(row["initial_trainables_SHA256"]) and sha(row["trainable_schema_SHA256"])
             and sha(row["initial_optimizer_SHA256"]) and sha(row["first_update_optimizer_SHA256"])
             and row["after_zero_forward_trainables_SHA256"] == row["initial_trainables_SHA256"]
             and row["after_zero_forward_optimizer_SHA256"] == row["initial_optimizer_SHA256"],
             f"{platform}: missing trainable/optimizer evidence or state changed in the zero-update forward")
        for key, count in (("zero_forward_inputs", 1), ("first_update_inputs", micros)):
            inputs = row[key]
            need(isinstance(inputs, list) and len(inputs) == count and all(sha(digest) for digest in inputs),
                 f"{platform}: missing ordered realized inputs, labels, masks or generation context")
        if identity["case"] in POLICY_CASES:
            need(sha(row["fixed_rollout_SHA256"]), f"{platform}: missing realized fixed rollout")
        if update_schedule is not None:
            second = row["second_update"]
            need(row["first_update_learning_rates"] == [0.0]
                 and row["first_update_trainables_SHA256"] == row["initial_trainables_SHA256"]
                 and row["first_update_optimizer_SHA256"] != row["initial_optimizer_SHA256"],
                 f"{platform}: first optimizer update must advance AdamW with zero LR and unchanged parameters")
            need(second["learning_rates"] == update_schedule[row["rank"]]
                 and second["before_trainables_SHA256"] == row["first_update_trainables_SHA256"]
                 and second["before_optimizer_SHA256"] == row["first_update_optimizer_SHA256"]
                 and sha(second["after_trainables_SHA256"]) and sha(second["after_optimizer_SHA256"])
                 and second["after_trainables_SHA256"] != second["before_trainables_SHA256"]
                 and second["after_optimizer_SHA256"] != second["before_optimizer_SHA256"]
                 and isinstance(second["inputs"], list) and len(second["inputs"]) == micros
                 and all(sha(digest) for digest in second["inputs"]),
                 f"{platform}: missing or inconsistent first parameter update evidence")
    return value


def verify_numerics(case, framework_commit, evidence, verify_raw):
    need(isinstance(case.get("numerics_plan"), dict)
         and case["numerics_plan"] == evidence.get("comparison_plan"),
         "numerics requires the registered case-specific comparison plan")
    plan, policy, rules = comparison_plan(case, verify_raw)
    identity = {"case": case["case"], "framework_commit": framework_commit,
                "original_launch_SHA256": case["launch_SHA256"]}
    shape = plan["training_ranks"], plan["microbatches_per_rank"]
    a100_identity = dict(identity, framework_commit=plan["A100_framework_commit"])
    if "reference_payload_sha256" in policy:
        need(sha(policy["reference_payload_sha256"]), "invalid original A100 reference payload identity")
        a100_identity["reference_payload_SHA256"] = policy["reference_payload_sha256"]
    cards = case["original_cards"]
    schedule = original_vlm_update_schedule(case, verify_raw) if case["case"] in ZERO_LR_VLM_REQUESTS else None
    a100 = receipt(evidence["A100"], a100_identity, shape, cards, "A100", verify_raw, schedule)
    hygon = receipt(evidence["Hygon"], identity, shape, cards, "Hygon", verify_raw, schedule)
    need(a100["run_id"] != hygon["run_id"] and a100["terminal_receipt_SHA256"] != hygon["terminal_receipt_SHA256"],
         "both platforms must have independent executions")
    same_keys = ["initial_trainables_SHA256", "trainable_schema_SHA256", "initial_optimizer_SHA256",
                 "zero_forward_inputs", "first_update_inputs"]
    if case["case"] in POLICY_CASES:
        same_keys.append("fixed_rollout_SHA256")
    for reference, target in zip(a100["ranks"], hygon["ranks"], strict=True):
        need(all(reference[key] == target[key] for key in same_keys),
             f"rank {reference['rank']}: realized inputs, rollout or initial state differ")
        if schedule is not None:
            need(reference["second_update"]["inputs"] == target["second_update"]["inputs"],
                 f"rank {reference['rank']}: realized second-update inputs differ")

    comparison = verify_raw(evidence["paired_comparison"])
    binding = dict(identity, plan_SHA256=evidence["comparison_plan"]["SHA256"],
                   A100_SHA256=evidence["A100"]["SHA256"], Hygon_SHA256=evidence["Hygon"]["SHA256"])
    if schedule is not None:
        binding["A100_update_schedule_SHA256"] = case["capture_value_evidence"]["SHA256"]
    need(all(comparison.get(key) == value for key, value in binding.items()),
         "paired comparison is not bound to the original policy and both execution receipts")
    failures = []
    for row in rank_values(comparison["ranks"], shape[0]):
        need(row["all_floating_tensors_finite"] is True and row["discrete_state_matches"] is True,
             f"rank {row['rank']}: nonfinite tensors or mismatched discrete state")
        if policy.get("metrics", {}).get("nonzero_total_gradient_and_update") is True:
            need(row.get("nonzero_gradient_and_update") is True,
                 f"rank {row['rank']}: zero gradient or update under the original policy")
        for phase, metrics in rules.items():
            observed = row[phase]
            need(set(observed) == set(metrics), f"rank {row['rank']}: incomplete {phase} metric inventory")
            for name, (direction, limit) in metrics.items():
                value = observed[name]
                need(finite_number(value), f"rank {row['rank']}: nonfinite comparison metric {name}")
                if (direction == "max" and value > limit) or (direction == "min" and value < limit):
                    failures.append((row["rank"], phase, name))
    need(not failures, f"original numerical thresholds failed: {failures}")
    return {"passed": True, "compared_ranks": shape[0]}
