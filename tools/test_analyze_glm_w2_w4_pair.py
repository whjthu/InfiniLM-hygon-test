"""CPU regressions for interpretation of a bounded, expensive GPU pair."""
import copy
import math
import unittest
from analyze_glm_w2_w4_pair import PLAN, paired_metric, summarize


def fixture():
    cases, rows = [], []
    for i, window in enumerate(PLAN):
        factor = 1 if window == 2 else .5
        cases.append({'index':i,'window':window,'warmup':i<2,'passed':True,'natural_EOS':True,
                      'request_id':str(i),'usage':{'prompt_tokens':8900,'completion_tokens':3},
                      'HTTP_wall_s':19*factor+3})
        calls = [{'past':j*512,'T':min(512,8900-j*512),'prefill':True,'Native_CAPI_s':factor}
                 for j in range(18)]
        calls += [{'past':8900+j,'T':1,'prefill':False,'Native_CAPI_s':.25} for j in range(2)]
        rows.append({'pair_index':i,'paired_warmup':i<2,'prefill_control_window':window,
                     'request_id':str(i),'prompt_tokens':8900,'requested_output_budget':4096,
                     'generated_tokens':3,'generated_token_ids_SHA256':'same-output',
                     'generated_natural_EOS':True,'reason':'eos_token',
                     'original_reset_all32_completed':True,'native_calls':calls,'wall_s':18*factor+1})
    return cases, rows


class InterpretationTests(unittest.TestCase):
    def test_warmups_cannot_create_gain(self):
        q = paired_metric([100000,.001,4,2,2,4])
        self.assertAlmostEqual(q['geometric_W2_over_W4_ratio'],2)
        self.assertFalse(q['statistical_significance_claim'])

    def test_log_linear_drift_cancels_within_one_block(self):
        q = paired_metric([1,1,2,4/2,8/2,16])
        self.assertAlmostEqual(q['geometric_W2_over_W4_ratio'],2)

    def test_changed_output_path_keeps_prefill_but_refuses_wall_attribution(self):
        cases, rows = fixture(); rows[3]['generated_token_ids_SHA256']='different'
        q = summarize(cases,rows)
        self.assertAlmostEqual(q['tail_prefill_CAPI_enclosure']['geometric_W2_over_W4_ratio'],2)
        self.assertFalse(q['matched_output_paths'])
        self.assertEqual(q['wall_schedule_attribution'],'CONFOUNDED_OUTPUT_PATH_OR_LENGTH')
        self.assertFalse(q['automatic_adoption'])

    def test_silent_prompt_omission_and_duplicate_decode_rejected(self):
        for mutation in ['omit_prefill','duplicate_decode']:
            cases, rows = fixture()
            if mutation == 'omit_prefill': rows[2]['native_calls'].pop(5)
            else: rows[2]['native_calls'][-1]['past']=8900
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): summarize(cases,rows)

    def test_wrong_arm_reset_or_eos_refuses_result(self):
        for key,value in [('prefill_control_window',1),('original_reset_all32_completed',False),
                          ('generated_natural_EOS',False),('reason','length')]:
            cases,rows=fixture(); rows[3][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): summarize(cases,rows)

    def test_invalid_or_overlapping_enclosure_rejected(self):
        for value in [float('nan'),-1,True,100000]:
            cases,rows=fixture(); rows[3]['native_calls'][0]['Native_CAPI_s']=value
            with self.subTest(value=value), self.assertRaises(ValueError): summarize(cases,rows)

    def test_incomplete_pair_cannot_report_ratio(self):
        cases,rows=fixture()
        with self.assertRaises(ValueError): summarize(cases[:-1],rows[:-1])


if __name__ == '__main__': unittest.main()
