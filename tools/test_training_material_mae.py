"""Test formal SF3D effect validation with original-byte fixture files."""

from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from training_formal_metrics import effect
from training_material_mae import MATERIAL_FIELDS, POLICY, scorer_sha256
from training_sample_artifacts import verifier


class MaterialEffectTest(unittest.TestCase):
    def setUp(self):
        private = TemporaryDirectory()
        self.addCleanup(private.cleanup)
        self.root = Path(private.name).resolve()
        self.ids = ["view-a", "view-b"]
        self.arms = {"A100": "a" * 64, "Hygon": "b" * 64, "Hygon_before": "c" * 64}
        self.identity = {"case": "T11.material-head", "framework_commit": "0" * 40,
                         "dataset_SHA256": "d" * 64, "scorer_SHA256": scorer_sha256()}
        self.documents = {}
        self.images = [self.artifact(sample, f"{sample}.rgba", sample.encode()) for sample in self.ids]
        self.queries = [self.artifact(sample, f"{sample}.f32",
                                      struct.pack("<6f", 0.25, 0, 0.5, 0.5, 0.25, 0))
                        for sample in self.ids]
        self.labels = [self.record(sample, pin, "reference", None, 0.25)
                       for sample, pin in zip(self.ids, self.images, strict=True)]
        self.predictions = {
            arm: [self.record(sample, pin, arm, self.arms[arm], base + (0.25 if i == 1 else 0.0))
                  for i, (sample, pin) in enumerate(zip(self.ids, self.images, strict=True))]
            for arm, base in (("A100", 0.25), ("Hygon", 0.25), ("Hygon_before", 0.50))
        }

    def artifact(self, sample, name, raw):
        (self.root / name).write_bytes(raw)
        return dict(sample_id=sample, status="scored", artifact=dict(location=name, SHA256=sha256(raw).hexdigest()))

    def record(self, sample, image, arm, checkpoint, level, *, albedo=None):
        if albedo is None:
            channels = [level, 0.25, level]
            albedo = [channels, channels] if checkpoint else channels
        value = dict(sample_id=sample, status="scored", RGBA_SHA256=image["artifact"]["SHA256"],
                     query_SHA256=self.queries[self.ids.index(sample)]["artifact"]["SHA256"],
                     checkpoint_SHA256=checkpoint,
                     material=dict(albedo=albedo, roughness=[level], metallic=[level]))
        return self.artifact(sample, f"{sample}-{arm}.json", json.dumps(value, sort_keys=True).encode())

    def put(self, doc):
        pin = sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
        self.documents[pin] = doc
        return {"SHA256": pin}

    def metric_row(self, metric, predictions=None, scores=None):
        if predictions is None:
            predictions = self.predictions
        identity = dict(self.identity, metric=metric)
        plan = self.put(dict(identity, sample_ids=self.ids, material_policy=POLICY,
                             material_RGBA_artifacts=self.images, material_references=self.labels,
                             material_query_artifacts=self.queries, query_points_per_sample=2))
        inventory = self.put(dict(identity, sample_plan_SHA256=plan["SHA256"],
                                  checkpoint_SHA256=self.arms, arms=predictions))
        weight = 2 / 3 if metric == "albedo MAE" else 1
        values = dict(A100=0.125 * weight, Hygon=0.125 * weight, Hygon_before=0.375 * weight)
        row = dict(values if scores is None else scores, planned_samples=2, sample_plan=plan,
                   sample_inventory=inventory)
        for arm, records in predictions.items():
            row[arm + "_failed"] = sum(record["status"] == "failed" for record in records)
            row[arm + "_scored"] = len(records) - row[arm + "_failed"]
        return dict(row, raw_scores=self.put(dict(row, **identity, checkpoint_SHA256=self.arms)),
                    paired_checkpoint_binding=self.put(dict(identity, **self.arms)))

    def verify(self, rows):
        return effect(rows, {metric: "lower" for metric in rows}, self.identity,
                      lambda x: self.documents[x["SHA256"]], verify_artifacts=verifier(self.root))

    def test_recomputes_all_three_metrics_using_the_same_original_records(self):
        rows = {metric: self.metric_row(metric) for metric in MATERIAL_FIELDS}
        self.assertEqual(self.verify(rows), {metric: {"absolute_degradation": 0, "relative_degradation": 0}
                                            for metric in MATERIAL_FIELDS})

    def test_rejects_declared_score_even_with_self_consistent_summary(self):
        metric = "roughness MAE"
        row = self.metric_row(metric, scores=dict(A100=0.125, Hygon=0, Hygon_before=0.375))
        with self.assertRaisesRegex(ValueError, "verified head outputs"):
            self.verify({metric: row})

    def test_rejects_missing_material_predictions_retaining_original_counts(self):
        metric = "metallic MAE"
        failed = [dict(self.predictions["Hygon"][0], status="failed"), self.predictions["Hygon"][1]]
        row = self.metric_row(metric, self.predictions | {"Hygon": failed})
        with self.assertRaisesRegex(ValueError, "requires all planned predictions"):
            self.verify({metric: row})

    def test_rejects_wrong_checkpoint_and_changed_original_rgba(self):
        metric = "roughness MAE"
        _, b = self.predictions["Hygon"]
        bad = self.record(self.ids[0], self.images[0], "other", self.arms["A100"], 0.25)
        row = self.metric_row(metric, self.predictions | {"Hygon": [bad, b]})
        with self.assertRaisesRegex(ValueError, "record identity"):
            self.verify({metric: row})
        (self.root / "view-a.rgba").write_bytes(b"changed original")
        with self.assertRaisesRegex(ValueError, "artifact body hash"):
            self.verify({metric: row})

    def test_rejects_metric_specific_prediction_replacement(self):
        first, second = "albedo MAE", "metallic MAE"
        _, b = self.predictions["Hygon"]
        clone = self.record(self.ids[0], self.images[0], "another-copy", self.arms["Hygon"], 0.25)
        rows = {first: self.metric_row(first),
                second: self.metric_row(second, self.predictions | {"Hygon": [clone, b]})}
        with self.assertRaisesRegex(ValueError, "different plans, scorers or head outputs"):
            self.verify(rows)

    def test_albedo_keeps_each_query_before_the_absolute_difference(self):
        metric = "albedo MAE"
        changed = self.record(self.ids[0], self.images[0], "special", self.arms["Hygon"], 0.25,
                              albedo=[[0, 0.25, 0], [0.5, 0.25, 0.5]])
        results = self.predictions | {"Hygon": [changed, self.predictions["Hygon"][1]]}
        false_scalar = self.metric_row(metric, results)
        with self.assertRaisesRegex(ValueError, "verified head outputs"):
            self.verify({metric: false_scalar})

    def test_rejects_missing_real_query_position(self):
        metric = "albedo MAE"
        self.queries[0] = self.artifact(self.ids[0], "incomplete.f32", struct.pack("<3f", 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "query count differs"):
            self.verify({metric: self.metric_row(metric)})
