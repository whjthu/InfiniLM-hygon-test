"""Synthetic CPU metrology checks; not Native, GPU or performance evidence."""
import math
import unittest
from emitter_timing_r244 import snapshot_emitter_timing as snapshot


def capture(times, *, eos=True, accepted=0.0, started=0.0):
    return snapshot(accepted=accepted, started=started, token_times=times,
                    generated_count=len(times), ends_in_eos=eos)


class EmitterTimingChecks(unittest.TestCase):
    def test_eos_exclusion_and_decode_off_by_one(self):
        q=capture([10.5,10.75,11.0,11.25],accepted=9.0,started=10.0)
        self.assertTrue(q['timestamps_valid'])
        self.assertEqual(q['model_non_eos_tokens'],3)
        self.assertEqual(q['non_eos_token_monotonic'],[10.5,10.75,11.0])
        self.assertEqual(q['user_decode_tok_s'],4.0)
        self.assertEqual(q['server_admitted_to_first_non_eos_s'],1.5)
        self.assertEqual(q['backend_started_to_first_non_eos_s'],.5)
        self.assertFalse(q['client_incremental_latency_qualified'])

    def test_per_request_percentile_convention(self):
        q=capture([0.,1.,3.,6.,10.,100.])
        self.assertAlmostEqual(q['ITL_P95_s'],3.85)
        self.assertAlmostEqual(q['ITL_P99_s'],3.97)
        self.assertAlmostEqual(q['user_decode_tok_s'],.4)

    def test_zero_one_and_no_eos_cases(self):
        for times in [[],[1.]]:
            q=capture(times,eos=False)
            self.assertTrue(q['timestamps_valid'])
            self.assertIsNone(q['user_decode_tok_s'])
        q=capture([1.]);self.assertEqual(q['model_non_eos_tokens'],0)
        self.assertIsNone(q['server_admitted_to_first_non_eos_s'])
        q=capture([1.,2.,4.],eos=False)
        self.assertAlmostEqual(q['user_decode_tok_s'],2/3)

    def test_equal_timestamp_burst_is_not_division_by_zero(self):
        q=capture([1.,1.,1.,2.])
        self.assertTrue(q['timestamps_valid'])
        self.assertFalse(q['rate_available'])
        self.assertIsNone(q['user_decode_tok_s'])
        self.assertEqual(q['ITL_P99_s'],0.)

    def test_bad_metadata_is_reported_without_runtime_exception(self):
        bad=[dict(accepted=None),dict(accepted=2.,started=1.),dict(started=5.),
             dict(token_times=None),dict(token_times=[1.,float('nan')]),
             dict(token_times=[1.,float('inf')]),dict(token_times=[2.,1.]),
             dict(token_times=[1.,True]),dict(generated_count=3),
             dict(generated_count=True),dict(accepted=10**1000),
             dict(accepted=-1e308,started=-1e308,token_times=[-1e308,1e308],ends_in_eos=False)]
        for patch in bad:
            args=dict(accepted=0.,started=0.,token_times=[1.,2.],generated_count=2,ends_in_eos=True);args.update(patch)
            with self.subTest(keys=list(patch)):
                q=snapshot(**args);self.assertFalse(q['timestamps_valid'])
                self.assertEqual(q['error'],'invalid_or_missing_timing_metadata')


if __name__=='__main__':
    unittest.main()
