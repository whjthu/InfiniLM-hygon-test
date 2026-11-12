"""Local metadata-only guard fixtures,not Native or repeatability evidence."""
import copy,math,unittest
from alias_policy import SCHEDULE,expected_counts
from alias_summary import summarize


class SummaryTests(unittest.TestCase):
    def data(self):
        trace=[dict(kind='prefill',past=0,tokens=1024),dict(kind='prefill',past=1024,tokens=218),
               dict(kind='decode_summary',first_past=1242,last_total=1245,calls=3)]
        owners=[dict(r454_index=i,Native_continuation=trace,history_alias=dict(mode=0,counters=[expected_counts(trace,0)]*8)) for i in range(2)]
        owners[1]['history_alias']['logits_audit']=[dict(input_SHA256='a'*64,first_logits_SHA256='b'*64,repeat_logits_SHA256='c'*64,differing_BF16_values=4,max_absolute=.03125,rms=.001) for _ in range(2)]
        results=[dict(case_id='one-fixed-gsm',native_forwards=5,generated_IDs_SHA256=str(i)*64,score=dict(correct=False,natural_EOS=True),backend_non_eos_decode_tok_s=2+i*.1) for i in range(2)]
        return owners,results

    def test_variation_is_recorded_without_fake_performance_or_quality(self):
        a,b=self.data();r=summarize(a,b)
        self.assertEqual(SCHEDULE,((0,0),(0,0)));self.assertEqual(r['requests'],2);self.assertEqual(r['correct'],0)
        self.assertFalse(r['generated_ids_match_in_audit_pair']);self.assertEqual(r['valid_timing_pairs'],0);self.assertFalse(r['promote'])

    def test_missing_request_is_rejected(self):
        a,b=self.data()
        with self.assertRaises(ValueError):summarize(a,b[:-1])

    def test_on_mode_or_alias_counters_are_rejected(self):
        for mode in (True,False):
            a,b=self.data()
            if mode:a[0]['history_alias']['mode']=1
            else:a[0]['history_alias']['counters'][0][2]=1
            with self.assertRaises(ValueError):summarize(a,b)

    def test_missing_or_nonfinite_prefill_audit_is_rejected(self):
        for missing in (True,False):
            a,b=self.data()
            if missing:a[1]['history_alias']['logits_audit']=a[1]['history_alias']['logits_audit'][:1]
            else:a[1]['history_alias']['logits_audit'][0]['rms']=math.nan
            with self.assertRaises(ValueError):summarize(a,b)


if __name__=='__main__':unittest.main()
