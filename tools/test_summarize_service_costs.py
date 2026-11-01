import copy
import unittest
from summarize_service_costs import MODELS, summarize


def report():
    row = {'admitted_index': 0, 'generated_token_ids': [10, 11, 1],
           'backend_token_monotonic': [12., 14., 15.], 'actual_generated_tokens': 3,
           'admission': {'accepted_monotonic': 10.}, 'started_monotonic': 11., 'ended_monotonic': 16.,
           'normal_reset_before': True, 'normal_reset_after': True, 'prompt_tokens': 20,
           'max_new_tokens': 4096, 'native_finish_reason': 'eos_token', 'final_metadata': {},
           'Native_forward_host_summary': {'prefill_s': 1., 'decode_s': 3.}}
    return {'actual_step': '910374.1218', 'source_manifest_SHA256': 'm',
            'Native_SHA256': MODELS['Deep']['native'], 'TP': 8, 'PP': 1, 'cards': 8,
            'completed': False, 'requests': [row]}


class CostsTest(unittest.TestCase):
    def run_summary(self, q):
        return summarize(q, 'Deep', '910374.1218', 'm')

    def test_eos_exclusion_and_interval_formula(self):
        q = self.run_summary(report())
        self.assertEqual(q['requests'][0]['visible_tokens'], 2)
        self.assertEqual(q['requests'][0]['backend_emitter_decode_tok_s'], .5)
        self.assertEqual(q['requests'][0]['backend_emitter_ITL_p95_s'], 2.)
        self.assertFalse(q['client_incremental_metrics_qualified'])

    def test_inflight_row_is_pending_not_success_or_failure(self):
        q = report()
        active = copy.deepcopy(q['requests'][0])
        active.update(admitted_index=1)
        del active['ended_monotonic']
        q['requests'].append(active)
        out = self.run_summary(q)
        self.assertEqual(out['completed_requests_with_checked_timelines'], 1)
        self.assertEqual(out['pending_requests'], [{'admitted_index': 1, 'prompt_tokens': 20, 'reported_error': False}])

    def test_bad_clock_and_token_count_rejected(self):
        for change in [{'backend_token_monotonic': [12., 11., 15.]}, {'actual_generated_tokens': 4}]:
            q = report()
            q['requests'][0].update(change)
            with self.assertRaises(ValueError):
                self.run_summary(q)

    def test_wrong_identity_and_topology_rejected(self):
        for change in [{'actual_step': 'other'}, {'source_manifest_SHA256': 'other'}, {'cards': 16}]:
            q = report()
            q.update(change)
            with self.assertRaises(ValueError):
                self.run_summary(q)

    def test_native_enclosure_over_wall_rejected(self):
        q = report()
        q['requests'][0]['Native_forward_host_summary']['decode_s'] = 20.
        with self.assertRaises(ValueError):
            self.run_summary(q)

    def test_single_token_cap_has_no_decode_rate(self):
        q = report()
        q['requests'][0].update(generated_token_ids=[10], backend_token_monotonic=[12.],
                                actual_generated_tokens=1, max_new_tokens=1, native_finish_reason='length')
        out = self.run_summary(q)
        self.assertIsNone(out['requests'][0]['backend_emitter_decode_tok_s'])
        self.assertTrue(out['requests'][0]['controlled_or_unexpected_cap'])


if __name__ == '__main__':
    unittest.main()
