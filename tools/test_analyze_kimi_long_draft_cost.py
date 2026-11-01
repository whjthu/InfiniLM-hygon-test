import unittest,json,sys,copy,hashlib
from pathlib import Path
from analyze_kimi_long_draft_cost import analyze
FIXTURE=json.loads(Path(sys.argv.pop(1)).read_bytes())
def rehash(q):
 w=q['runs']['Kimi']['watch'];w['report_SHA256']=hashlib.sha256((json.dumps(w['report'],indent=2)+'\n').encode()).hexdigest()
class CostAudit(unittest.TestCase):
 def test_actual_completed(self):self.assertEqual(analyze(FIXTURE)['decision'],'COST_DOES_NOT_EXCLUDE_CONDITIONAL_CAPTURE_AND_ACCEPTANCE_CARD')
 def test_partial_queries(self):
  q=copy.deepcopy(FIXTURE);q['runs']['Kimi']['watch']['report']['cases'][2]['samples'].pop();rehash(q)
  with self.assertRaises(ValueError):analyze(q)
 def test_post_KFD_busy(self):
  q=copy.deepcopy(FIXTURE);q['runs']['Kimi']['watch']['post']['kfd_pids']=['123']
  with self.assertRaises(ValueError):analyze(q)
 def test_nonfinite_clock(self):
  q=copy.deepcopy(FIXTURE);q['runs']['Kimi']['watch']['report']['cases'][2]['samples'][0]['complete_query_D2H_wall_s']=float('nan');rehash(q)
  with self.assertRaises(ValueError):analyze(q)
 def test_expanded_fixture_cannot_be_acceptance(self):
  q=copy.deepcopy(FIXTURE);q['runs']['Kimi']['watch']['report']['target_acceptance_qualified']=True;rehash(q)
  with self.assertRaises(ValueError):analyze(q)
 def test_hard_runtime(self):
  q=copy.deepcopy(FIXTURE);next(x for x in q['runs']['Kimi']['accounts']if x[0]=='910371.601')[4]='00:10:01'
  with self.assertRaises(ValueError):analyze(q)
if __name__=='__main__':unittest.main()
