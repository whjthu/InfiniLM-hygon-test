"""Synthetic numerical/refusal checks; never evidence of a Native/GPU run."""
import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pair_analysis', pathlib.Path(__file__).with_name('analyze_glm_prefill_pair.py'))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def fixture():
    cases, rows = [], []
    for i, window in enumerate(a.PLAN):
        calls = [{'prefill': True, 'past': n * 512, 'T': min(512, 8900 - n * 512), 'Native_CAPI_s': 1.0}
                 for n in range(18)] + [{'prefill': False, 'past': 8900, 'T': 1, 'Native_CAPI_s': 1.0}]
        rows.append({'pair_index': i, 'prefill_control_window': window, 'paired_warmup': i < 2,
                     'request_id': 'SYNTHETIC-' + str(i), 'prompt_tokens': 8900, 'requested_output_budget': 4096,
                     'generated_tokens': 2, 'generated_natural_EOS': True, 'reason': 'eos_token',
                     'original_reset_all32_completed': True, 'actual_prompt_token_ids_SHA256': 'f' * 64,
                     'generated_token_ids_SHA256': 'e' * 64, 'native_calls': calls, 'wall_s': 20.0})
        cases.append({'index': i, 'window': window, 'warmup': i < 2, 'passed': True,
                      'request_id': rows[-1]['request_id'], 'usage': {'prompt_tokens': 8900, 'completion_tokens': 2}, 'HTTP_wall_s': 21.0})
    return cases, rows


class PairAnalysisChecks(unittest.TestCase):
    def test_neutral_and_known_speedup(self):
        self.assertAlmostEqual(a.paired_metric([100.0] * 18)['geometric_W1_over_W2_ratio'], 1)
        values = [90000.0, .0001] + [20.0 if w == 1 else 10.0 for w in a.PLAN[2:]]
        out = a.paired_metric(values)
        self.assertAlmostEqual(out['geometric_W1_over_W2_ratio'], 2)
        self.assertAlmostEqual(out['candidate_time_reduction_percent'], 50)
        self.assertEqual(out['blocks_faster_W2'], 4)
        self.assertFalse(out['statistical_significance_claim'])

    def test_log_linear_drift_cancels(self):
        values = [math.exp(.02 * i) * (20 if w == 1 else 10) for i, w in enumerate(a.PLAN)]
        out = a.paired_metric(values)
        for block in out['blocks']:
            self.assertAlmostEqual(block['W1_over_W2_geometric_ratio'], 2)

    def test_inconsistent_blocks_remain_visible(self):
        out = a.paired_metric([1, 1] + [2, 1, 1, 2] * 3 + [1, 2, 2, 1])
        self.assertEqual(out['blocks_faster_W2'], 3)
        self.assertLess(out['block_ratio_min'], 1)
        self.assertIsNone(out['confidence_interval'])

    def test_output_length_confound_not_adopted(self):
        cases, rows = fixture()
        rows[3]['generated_tokens'] = cases[3]['usage']['completion_tokens'] = 3
        rows[3]['native_calls'].append({'prefill': False, 'past': 8901, 'T': 1, 'Native_CAPI_s': .5})
        out = a.summarize(cases, rows)
        self.assertFalse(out['matched_measured_output_lengths'])
        self.assertEqual(out['wall_schedule_attribution'], 'CONFOUNDED_OUTPUT_LENGTH')
        self.assertFalse(out['automatic_adoption'])

    def test_invalid_geometry_order_and_timing_rejected(self):
        for change in ['geometry', 'order', 'nan', 'missing']:
            cases, rows = fixture()
            if change == 'geometry': rows[2]['native_calls'][17]['T'] -= 1
            if change == 'order': rows[3]['pair_index'] = 2
            if change == 'nan': rows[2]['native_calls'][0]['Native_CAPI_s'] = float('nan')
            if change == 'missing': rows.pop()
            with self.subTest(change=change), self.assertRaises(ValueError): a.summarize(cases, rows)

    def test_all32_archive_hash_and_terminal_refusal(self):
        with tempfile.TemporaryDirectory(prefix='glm-pair-SYNTHETIC-') as folder:
            root = pathlib.Path(folder); cases, rows = fixture(); run = 'SYNTHETIC-ONLY'; handle = 'SYNTHETIC.0'
            def write(relative, value):
                p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
                raw = json.dumps(value, allow_nan=False).encode(); p.write_bytes(raw)
                return hashlib.sha256(raw).hexdigest()
            owners = []
            for stage in range(4):
                for rank in range(8):
                    path = pathlib.Path('results') / ('SYNTHETIC-owner-%d-%d.json' % (stage, rank))
                    q = dict(stage=stage, rank=rank, paired_run=run, paired_actual_step=handle, library_sha256=a.NATIVE,
                             paired_source_binding_r241={'source_manifest_SHA256': a.MANIFEST, 'all_source_files_bound_before_Torch': True}, service_requests=rows)
                    q.update({k: True for k in ['completed', 'normal_shutdown', 'normal_reset_close', 'worker_shutdown_completed',
                                               'native_service_loop_completed', 'post_close_Torch_liveness', 'normal_HTTP_shutdown']})
                    digest = write(path, q)
                    owners.append({'stage': stage, 'rank': rank, 'path': str(a.REMOTE / path.as_posix()), 'SHA256': digest})
            client = dict(completed=True, passed=True, no_retries=True, attempts=18, completed_HTTP_200=18,
                          run=run, actual_GPU_handle=handle, request_file_SHA256=a.REQUEST, cases=cases)
            prefix = pathlib.Path('results') / (a.LABEL + '-client-' + run)
            watch = dict(completed=True, post_empty=True, Native_lifecycle_qualified=True, paired_measurement_conditions_qualified=True,
                         restarts=0, source_manifest_SHA256=a.MANIFEST, Native_SHA256=a.NATIVE, actual_handle=handle, run=run,
                         terminal=[handle, run, 'COMPLETED', '0:0'], nodes=['n0', 'n1', 'n2', 'n3'],
                         post=[{'node': 'n' + str(i), 'KFD_empty': True, 'port43843_bindable': True} for i in range(4)],
                         network_and_normal=[{'stage': s, 'normal_markers': 1, 'IBext_v8_mentions': 1, 'Socket_network_mentions': 0} for s in range(4)],
                         owner_results=owners, paired_client_result_SHA256=write(prefix / 'result.json', client),
                         paired_wrapper_SHA256=write(str(prefix) + '-wrapper.json', {'completed': True, 'passed': True, 'single_HTTP_child_reaped': True}))
            write(a.LABEL + '-watch.json', watch)
            self.assertAlmostEqual(a.analyze(root)['HTTP_wall']['geometric_W1_over_W2_ratio'], 1)
            p = root / 'results/SYNTHETIC-owner-0-0.json'; original = p.read_bytes(); p.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): a.analyze(root)
            p.write_bytes(original)
            bad = copy.deepcopy(watch); bad['owner_results'].pop(); write(a.LABEL + '-watch.json', bad)
            with self.assertRaisesRegex(ValueError, 'All32'): a.analyze(root)
            bad = copy.deepcopy(watch); bad['completed'] = False; write(a.LABEL + '-watch.json', bad)
            with self.assertRaisesRegex(ValueError, 'Terminal qualified'): a.analyze(root)


if __name__ == '__main__':
    unittest.main()
