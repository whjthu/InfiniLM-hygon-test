"""Regression checks for the paired first-update acceptance gate."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from training_numerics import original_vlm_update_schedule, verify_numerics


ROOT = Path(__file__).resolve().parents[1]
TARGET_COMMIT = "1" * 40
A100_COMMIT = "2" * 40
LAUNCH = "3" * 64
UPDATE_RULES = {
    "prediction_l2": ("prediction", "prediction_relative_l2_max"),
    "loss_rel": ("loss", "loss_relative_error_max"),
    "pre_l2": ("gradient_before_clip", "gradient_preclip_relative_l2_max"),
    "pre_cos": ("gradient_before_clip", "gradient_preclip_cosine_min"),
    "pre_norm_rel": ("gradient_before_clip", "gradient_preclip_norm_relative_error_max"),
    "post_l2": ("gradient_after_clip", "gradient_postclip_relative_l2_max"),
    "update_l2": ("parameter_update", "parameter_update_relative_l2_max"),
    "exp_avg": ("optimizer_state", "optimizer_moment_relative_l2_max"),
    "exp_avg_sq": ("optimizer_state", "optimizer_moment_relative_l2_max"),
}


class NumericsTest(unittest.TestCase):
    def setUp(self):
        self.docs = {}
        self.case = dict(case="T06.sft", launch_SHA256=LAUNCH, original_cards=1)

        source = ROOT / "code/bw1000/training-sdxl-A100-operand-provenance-v1/policy.json"
        self.policy = json.loads(source.read_bytes())
        policy_pin = self.add("policy", "a", self.policy)
        original = self.add("original", "f", dict(case=self.case["case"], framework_commit=A100_COMMIT,
                            reference_SHA256=self.policy["reference_payload_sha256"],
                            policy_SHA256=policy_pin["SHA256"]))
        rules = {}
        for name, (category, path) in UPDATE_RULES.items():
            rules[name] = dict(category=category, direction=path.rsplit("_", 1)[-1],
                               policy_path=["metrics", path])
            if category == "optimizer_state":
                rules[name]["optimizer_state_component"] = name
        for path, value in self.policy["posterior_additional_metrics"].items():
            if type(value) in (int, float):
                rules[path] = dict(category="generation_state", direction="max",
                                   policy_path=["posterior_additional_metrics", path])
        self.plan = dict(case="T06.sft", original_launch_SHA256=LAUNCH, A100_framework_commit=A100_COMMIT,
                         training_ranks=1, microbatches_per_rank=1, original_policy=policy_pin,
                         original_reference=original,
                         metrics=dict(before_update={"prediction_l2": rules["prediction_l2"].copy()},
                                      after_update=rules))
        plan = self.add("plan", "b", self.plan)
        self.case["numerics_plan"] = plan

        rank = dict(rank=0, initial_trainables_SHA256="4" * 64, trainable_schema_SHA256="5" * 64,
                    initial_optimizer_SHA256="6" * 64, after_zero_forward_trainables_SHA256="4" * 64,
                    after_zero_forward_optimizer_SHA256="6" * 64, first_update_optimizer_SHA256="7" * 64,
                    zero_forward_inputs=["8" * 64], first_update_inputs=["9" * 64])
        identity = dict(case=self.case["case"], original_launch_SHA256=LAUNCH, framework_commit=TARGET_COMMIT)
        receipt = dict(identity, terminal_status="COMPLETED", exit_code=0, cards=1,
                       optimizer_update_counts=dict(zero_forward=0, first_update=1), ranks=[rank])
        self.reference = deepcopy(dict(receipt, platform="A100", framework_commit=A100_COMMIT,
                                       reference_payload_SHA256=self.policy["reference_payload_sha256"],
                                       run_id="a" * 64, terminal_receipt_SHA256="b" * 64))
        self.target = deepcopy(dict(receipt, platform="Hygon", run_id="c" * 64, terminal_receipt_SHA256="d" * 64))
        a100 = self.add("a100", "c", self.reference)
        hygon = self.add("hygon", "d", self.target)
        valid = {name: self.policy[rule["policy_path"][0]][rule["policy_path"][1]]
                 for name, rule in rules.items()}
        self.comparison = dict(identity, plan_SHA256=plan["SHA256"], A100_SHA256=a100["SHA256"],
                               Hygon_SHA256=hygon["SHA256"],
                               ranks=[dict(rank=0, all_floating_tensors_finite=True, discrete_state_matches=True,
                                           nonzero_gradient_and_update=True,
                                           before_update={"prediction_l2": valid["prediction_l2"]},
                                           after_update=valid)])
        evidence = self.add("comparison", "e", self.comparison)
        self.evidence = dict(passed=True, comparison_plan=plan, A100=a100, Hygon=hygon, paired_comparison=evidence)

    def add(self, name, digest, doc):
        pin = dict(path=name, SHA256=digest * 64)
        self.docs[name] = pin, doc
        return pin

    def read(self, pin):
        expected, doc = self.docs[pin["path"]]
        self.assertEqual(pin, expected)
        return doc

    def check(self):
        return verify_numerics(self.case, TARGET_COMMIT, self.evidence, self.read)

    def test_accepts_original_limits_and_distinct_a100_revision(self):
        self.assertEqual(self.check(), {"passed": True, "compared_ranks": 1})

    def test_pass_flag_alone_is_rejected(self):
        self.evidence = {"passed": True}
        with self.assertRaisesRegex(ValueError, "registered case-specific"):
            self.check()

    def test_actual_sdxl_failed_measurement_cannot_be_relabelled(self):
        recorded = ROOT / "results/bw1000/2026-11-20/training-sdxl-original540-frozen-conv-dgrad-fp32-terminal.json"
        result = json.loads(recorded.read_bytes())
        value = result["receipts"]["comparison-summary.json"]["fields"]["metrics"]["parameter_update"]["relative_l2"]
        self.comparison["ranks"][0]["after_update"]["update_l2"] = value
        with self.assertRaisesRegex(ValueError, "original numerical thresholds failed.*update_l2"):
            self.check()

    def test_rejects_input_substitution(self):
        self.target["ranks"][0]["first_update_inputs"] = ["e" * 64]
        with self.assertRaisesRegex(ValueError, "realized inputs"):
            self.check()

    def test_rejects_optimizer_mutation_during_zero_update_pass(self):
        self.target["ranks"][0]["after_zero_forward_optimizer_SHA256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "state changed"):
            self.check()

    def test_rejects_a_missing_original_optimizer_moment(self):
        del self.plan["metrics"]["after_update"]["exp_avg_sq"]
        with self.assertRaisesRegex(ValueError, "original optimizer moment"):
            self.check()

    def test_rejects_omission_of_original_generation_threshold(self):
        del self.plan["metrics"]["after_update"]["unscaled_latent_relative_l2_max"]
        with self.assertRaisesRegex(ValueError, "omits original numerical thresholds"):
            self.check()

    def test_rejects_reversed_cosine_rule(self):
        self.plan["metrics"]["after_update"]["pre_cos"]["direction"] = "max"
        with self.assertRaisesRegex(ValueError, "direction changed"):
            self.check()

    def test_rejects_incomplete_multirank_capture(self):
        self.case["original_cards"] = self.plan["training_ranks"] = 2
        self.reference["cards"] = self.target["cards"] = 2
        with self.assertRaisesRegex(ValueError, "rank inventory"):
            self.check()

    def test_rejects_wrong_card_count(self):
        self.target["cards"] = 2
        with self.assertRaisesRegex(ValueError, "wrong launch, source or terminal evidence"):
            self.check()

    def test_requires_policy_update_metrics_for_rl(self):
        self.case["case"] = self.plan["case"] = "T06.ddpo"
        with self.assertRaisesRegex(ValueError, "metric categories are incomplete"):
            self.check()

    def test_rejects_a_missing_reward_component(self):
        self.case["case"] = self.plan["case"] = "T08.motion-rft"
        self.policy["policy_thresholds"] = {
            key: 0.01 for key in ("reward", "old_log_probability", "new_log_probability", "ratio", "advantage")
        }
        for key in self.policy["policy_thresholds"]:
            self.plan["metrics"]["after_update"][key] = dict(category=key, direction="max",
                                                               policy_path=["policy_thresholds", key])
        self.plan["reward_components"] = ["text", "motion"]
        self.plan["metrics"]["after_update"]["reward"]["reward_component"] = "text"
        with self.assertRaisesRegex(ValueError, "omits original reward components"):
            self.check()

    def configure_vlm_warmup(self, case_id="T04.sft"):
        matrix = json.loads((ROOT / "coordination/training/training-case-acceptance-matrix-v1.json").read_bytes())
        original = next(row for row in matrix["cases"] if row["case"] == case_id)
        pin = original["capture_value_evidence"]
        self.docs[pin["path"]] = pin, json.loads((ROOT / pin["path"]).read_bytes())
        self.case.update(case=case_id, original_cards=8, capture_value_evidence=pin)
        self.plan.update(case=case_id, training_ranks=8, microbatches_per_rank=4)
        self.docs["original"][1]["case"] = case_id
        self.plan["metrics"]["after_first_parameter_update"] = deepcopy(self.plan["metrics"]["after_update"])
        for platform in (self.reference, self.target):
            platform.update(case=case_id, cards=8)
            platform["optimizer_update_counts"]["first_parameter_update"] = 2
            row = platform["ranks"][0]
            row.update(first_update_inputs=["9" * 64] * 4, first_update_learning_rates=[0.0],
                       first_update_trainables_SHA256=row["initial_trainables_SHA256"],
                       second_update=dict(learning_rates=[0.0001], inputs=["a" * 64] * 4,
                                          before_trainables_SHA256=row["initial_trainables_SHA256"],
                                          before_optimizer_SHA256=row["first_update_optimizer_SHA256"],
                                          after_trainables_SHA256="b" * 64, after_optimizer_SHA256="c" * 64))
            platform["ranks"] = [dict(deepcopy(row), rank=i) for i in range(8)]
        row = self.comparison["ranks"][0]
        row["after_first_parameter_update"] = deepcopy(row["after_update"])
        self.comparison.update(case=case_id, A100_update_schedule_SHA256=pin["SHA256"],
                               ranks=[dict(deepcopy(row), rank=i) for i in range(8)])

    def test_reads_both_real_original_zero_lr_schedules(self):
        for case in ("T03.sft", "T04.sft"):
            with self.subTest(case=case):
                self.configure_vlm_warmup(case)
                self.assertEqual(original_vlm_update_schedule(self.case, self.read), [[0.0001]] * 8)
                self.assertEqual(self.check()["compared_ranks"], 8)

    def test_vlm_rejects_zero_lr_only_and_second_step_input_substitution(self):
        self.configure_vlm_warmup()
        del self.target["optimizer_update_counts"]["first_parameter_update"]
        with self.assertRaisesRegex(ValueError, "must be optimizer step 2"):
            self.check()
        self.target["optimizer_update_counts"]["first_parameter_update"] = 2
        self.target["ranks"][3]["second_update"]["inputs"][2] = "d" * 64
        with self.assertRaisesRegex(ValueError, "rank 3: realized second-update inputs differ"):
            self.check()

    def test_vlm_rejects_optimizer_discontinuity_or_no_parameter_change(self):
        self.configure_vlm_warmup()
        row = self.target["ranks"][1]
        second = row["second_update"]
        second["before_optimizer_SHA256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "inconsistent first parameter update"):
            self.check()
        second["before_optimizer_SHA256"] = row["first_update_optimizer_SHA256"]
        second["after_trainables_SHA256"] = second["before_trainables_SHA256"]
        with self.assertRaisesRegex(ValueError, "inconsistent first parameter update"):
            self.check()

    def test_vlm_applies_original_thresholds_at_the_first_parameter_update(self):
        self.configure_vlm_warmup()
        self.comparison["ranks"][5]["after_first_parameter_update"]["update_l2"] = 0.6
        with self.assertRaisesRegex(ValueError, "5, 'after_first_parameter_update', 'update_l2'"):
            self.check()


if __name__ == "__main__":
    unittest.main()
