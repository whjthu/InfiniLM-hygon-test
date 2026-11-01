import copy
import unittest
from analyze_kimi_c1_delta_profile import summarize


def fixture():
    workers=[]
    for rank in range(16):
        samples=[]
        for arm in ['OFF-before','ON','OFF-after']:
            for t in range(10):
                samples.append({'arm':arm,'trial':t,'warmup':t<2,
                    'API_wall_ms':1000 if t<2 else 1+(rank/1000)+(.1 if arm=='ON'else 0),
                    'output_state_equal_to_OFF':True,'source_rows_unchanged':True,
                    'profile':{'C':1,'T':1,'pool':3,'added_segment_syncs':0,
                        'stream_total_ms':.8,'segments':[{'name':'QKV','stream_elapsed_ms':.3},
                        {'name':'KDA','stream_elapsed_ms':.5}]}if arm=='ON'else None})
        workers.append({'rank':rank,'completed':True,'normal_close':True,'layer_calls':31,'samples':samples})
    return workers


class TimingInterpretationTests(unittest.TestCase):
    def test_warmup_and_max_owner(self):
        q=summarize(fixture())
        self.assertAlmostEqual(q['max16owner_APIwall_ms']['OFF-before']['median'],1.015)
        self.assertEqual(q['max16owner_APIwall_ms']['ON']['n'],8)

    def test_observer_overhead_is_not_speedup(self):
        q=summarize(fixture())
        self.assertGreater(q['APIwall_ON_over_OFF_bracket'],1.03)
        self.assertFalse(q['APIwall_absolute_bracket_within3pct'])
        self.assertFalse(q['new_kernel_or_Model_gain'])

    def test_missing_owner_or_failed_state_cannot_qualify(self):
        workers=fixture()
        with self.assertRaises(ValueError):summarize(workers[:-1])
        workers[0]['samples'][15]['source_rows_unchanged']=False
        with self.assertRaises(ValueError):summarize(workers)

    def test_uncovered_stream_interval_rejected(self):
        workers=fixture();workers[0]['samples'][12]['profile']['stream_total_ms']=1.0
        with self.assertRaises(ValueError):summarize(workers)


if __name__=='__main__':unittest.main()
