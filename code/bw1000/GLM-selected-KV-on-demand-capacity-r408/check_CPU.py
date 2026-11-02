from pathlib import Path
import json,hashlib,tempfile,ast
from consumer import recording_contract,consume,NativeLeaf,CASES,MODES,seed_chunks
from checkpoint_loader import check_recording
from journal import Journal,aggregate
def run():
 schedule=recording_contract();loader=check_recording();assert schedule['query_calls']==12 and schedule['seed_calls']==774
 D=Path(__file__).resolve().parent
 for p in D.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 helper=(D/'selected_kv_r408.hpp').read_text();caller=(D/'caller_selected_kv_r408.cpp').read_text();bridge=(D/'attention_component_r408.cpp').read_text()
 for t in [caller,bridge]:
  assert '#include "selected_kv_r408.hpp"'in t and '::append_latent_rotary('in t and '::store_projected('in t and '::selected('in t
 assert 'auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);'in caller
 assert 'auto fresh=c->prep->expand_cached(*c->r,a.kv_latent,a.k_rot);'in bridge
 assert 'held.insert(held.end(),{allK,allV});'in helper and helper.index('held.insert(held.end(),{allK,allV});')<helper.index('v_on_demand_r400::for_blocks')
 assert 'held.insert(held.end(),{kv.first,kv.second});allK->narrow'in helper
 assert 'r.syncStream()'not in helper,'New helper must not add a fence'
 class Leaf:
  def __init__(self,fail=False):self.closed=False;self.fail=fail
  def check_deadline(self):pass
  def begin(self,mode):pass
  def seed(self,past,count):pass
  def query(self,c,ids):return {'synthetic_recording_only':True,'Native_executed':False}
  def inspect(self,c,ids):
   if self.fail:raise RuntimeError('directK fail after saved primary')
   return {'finite':True,'Native_executed':False}
  def validate(self,r):pass
  def close(self):self.closed=True
 with tempfile.TemporaryDirectory()as temp:
  for fail in [False,True]:
   p=Path(temp)/('failure.jsonl'if fail else'normal.jsonl');journal=Journal(p);leaf=Leaf(fail)
   try:out=consume(leaf,journal.update)
   except RuntimeError:assert fail
   else:assert not fail and len(out)==12
   finally:journal.close()
   q=aggregate(p);assert leaf.closed and len(q['records'])==(1 if fail else 12)
   events=[json.loads(s)for s in p.read_text().splitlines()];assert len([e for e in events if e['type']=='query'])==(1 if fail else 12)
   if fail:assert q['errors']and q['records'][0]['K_stage_diagnostic_error']
   else:assert len(events)==14 and not q['errors'],'12query+2diagnostic only, no fullhistory rewrite'
 calls=[]
 class Lib:
  def glm_attention_destroy_r398(self,*a):calls.append(a);return 1
  def glm_attention_error_r398(self):return b'CPU destroy failure'
 leaf=NativeLeaf.__new__(NativeLeaf);leaf.lib=Lib();leaf.ctx=1;leaf.closed=False;leaf.destroy_attempted=False
 try:leaf.close()
 except RuntimeError:pass
 leaf.close();assert len(calls)==1
 return dict(passed=True,actual_consumer=schedule,actual_loader=loader,shared_pair_geometry_state_copy_source_checked=True,fresh_projection_retained=True,held_before_copy=True,no_added_sync=True,actual_O1_scalar_journal=True,diagnostic_failure_primary_retained=True,failed_destroy_not_reissued=True,Native_dlopen=False,GPU=0,weights_opened=False,Model_qualification=False)
if __name__=='__main__':print(json.dumps(run()))
