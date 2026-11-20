import json
from pathlib import Path
import unittest

from training_A100_engineering_timing import check_generation, check_vggt, check_vlm


ROOT = Path(__file__).resolve().parents[1]


class A100EngineeringTimingTest(unittest.TestCase):
    def setUp(self):
        matrix = json.loads((ROOT / 'coordination/training/training-case-acceptance-matrix-v1.json').read_bytes())
        self.cases = {case['case']: case for case in matrix['cases']}
        self.audits = {key: json.loads((ROOT / item['path']).read_bytes())
                       for key, item in matrix['original_A100_engineering_timing_audits'].items()}

    def test_recomputes_rates_on_all_three_original_audits(self):
        generation = check_generation(self.cases, self.audits['generation'])
        vlm = check_vlm(self.cases, self.audits['vlm'])
        vggt = check_vggt(self.cases['T12.camera-head'], self.audits['vggt'])
        self.assertEqual((len(generation), len(vlm), len(vggt)), (8, 5, 1))
        self.assertAlmostEqual(generation['T06.ddpo'], 0.38992760311297503)
        self.assertAlmostEqual(vlm['T03.sft'], 0.17327135683884817)
        self.assertAlmostEqual(vggt['T12.camera-head'], 0.47501553412383735)

    def test_rejects_rollouts_as_updates(self):
        audit = self.audits['generation']
        row = audit['cases']['T06.ddpo']['repeats'][0]
        row['updates_per_second'] = row['original_rate']
        with self.assertRaisesRegex(ValueError, 'normalized update rate differs'):
            check_generation(self.cases, audit)

    def test_rejects_replayed_original_window(self):
        audit = self.audits['generation']
        rows = audit['cases']['T11.material-head']['repeats']
        rows[1]['raw_step_log_SHA256'] = rows[0]['raw_step_log_SHA256']
        with self.assertRaisesRegex(ValueError, 'reused or invalid original timing record'):
            check_generation(self.cases, audit)

    def test_rejects_fast_rank_substitution(self):
        audit = self.audits['vlm']
        row = audit['cases']['T01.sft']['repeats'][0]
        row['slowest_rank'] = row['rank_window_seconds'].index(min(row['rank_window_seconds']))
        with self.assertRaisesRegex(ValueError, 'slowest rank differs'):
            check_vlm(self.cases, audit)

    def test_rejects_rank_receipt_reuse_even_if_final_state_matches(self):
        audit = self.audits['vlm']
        rows = audit['cases']['T04.sft']['repeats']
        self.assertEqual(rows[1]['final_trainer_state_SHA256'], rows[0]['final_trainer_state_SHA256'])
        rows[1]['rank_timing_SHA256'][0] = rows[0]['rank_timing_SHA256'][0]
        with self.assertRaisesRegex(ValueError, 'reused or invalid VLM timing record'):
            check_vlm(self.cases, audit)

    def test_rejects_vggt_kernel_time_as_complete_window(self):
        audit = self.audits['vggt']
        row = audit['runs'][0]
        row['full_measurement_seconds'] = row['recorded_step_sum_seconds']
        with self.assertRaisesRegex(ValueError, 'complete wall window'):
            check_vggt(self.cases['T12.camera-head'], audit)

    def test_rejects_vggt_result_from_another_repeat(self):
        audit = self.audits['vggt']
        audit['runs'][0]['original_result_SHA256'] = audit['runs'][1]['original_result_SHA256']
        with self.assertRaisesRegex(ValueError, 'original result SHA differs'):
            check_vggt(self.cases['T12.camera-head'], audit)


if __name__ == '__main__':
    unittest.main()
