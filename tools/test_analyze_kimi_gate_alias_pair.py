import unittest
from analyze_kimi_gate_alias_pair import summarize,PLAN


def fixture():
    out=[]
    for rank in range(16):
        samples=[]
        for i,(arm,b,warm)in enumerate(PLAN):
            samples.append({'sample_index':i,'arm':arm,'block':b,'warmup':warm,'API_wall_ms':
                100000 if warm else (4 if arm=='OFF'else 2)*(1+rank/100),
                'finite':True,'source_rows_unchanged':True,'profile_enabled':False,
                'actual_alias_counter_delta':[1,int(arm=='ON'),int(arm=='OFF')],
                'numeric_diagnostics':{k:{'relative_RMS':0.,'max_abs':0.}for k in ['output','conv_write_row2','SSM_write_row2']}})
        out.append({'rank':rank,'completed':True,'normal_close':True,'layer_calls':35,'samples':samples})
    return out


class PairTests(unittest.TestCase):
    def test_warmups_and_max16_owner(self):
        q=summarize(fixture());self.assertAlmostEqual(q['geometric_OFF_over_ON_ratio'],2)
        self.assertAlmostEqual(q['blocks'][0]['OFF_ms'][0],4.6)
        self.assertFalse(q['Model_gain_qualified']);self.assertFalse(q['statistical_significance_claim'])
    def test_no_selected_candidate_cannot_report_gain(self):
        w=fixture();w[4]['samples'][4]['actual_alias_counter_delta']=[1,0,1]
        with self.assertRaises(ValueError):summarize(w)
    def test_incomplete_block_rejected(self):
        w=fixture();w[0]['samples'].pop()
        with self.assertRaises(ValueError):summarize(w)
    def test_update_row_error_cannot_be_diluted_by_source_rows(self):
        w=fixture();w[0]['samples'][4]['numeric_diagnostics']['SSM_write_row2']['relative_RMS']=.03
        with self.assertRaises(ValueError):summarize(w)
    def test_nonlinear_off_drift_is_exposed(self):
        w=fixture()
        for owner in w:owner['samples'][5]['API_wall_ms']*=4
        q=summarize(w);self.assertGreater(q['max_OFF_relative_spread'],1)
        self.assertFalse(q['point_gain_exceeds_largest_OFF_pair_spread'])


if __name__=='__main__':unittest.main()
