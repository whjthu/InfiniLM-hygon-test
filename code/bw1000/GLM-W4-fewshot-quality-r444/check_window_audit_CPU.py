"""Changed actual control-flow consumer only. Recording leaves, no device/math claim."""
from pathlib import Path
import hashlib,importlib.util,json,sys
D=Path(__file__).resolve().parent
P=D.parent/'public-GLM-native-provider-r425/public_glm_native'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(provider_root=None):
 P=Path(provider_root)if provider_root is not None else D.parent/'public-GLM-native-provider-r425/public_glm_native'
 assert sha(P/'glm_prefill_execute_r236.py')=='1bcb37ddcb25593ed64ac9309749d013f07b9c469d426446e1a9211c53a4f402'
 execute=load('r444_original_actual_execute',P/'glm_prefill_execute_r236.py').execute_plan
 observed=load('r444_actual_observer',D/'prefill_window_audit.py').execute_observed
 def run(n,window,*,traced,cancel_vote=None,forward_error=None,prefill=True,discard_inside_sample=False):
  sizes=[512]*(n-1)+[17];plan=[];past=0
  for size in sizes:plan.append((past,list(range(size))));past+=size
  events=[];votes=0;forwards=0;record=[];injected=RuntimeError('recording Native leaf failure')
  def forward(p,values,final):
   nonlocal forwards
   assert values is plan[forwards][1];events.append(('forward',p,len(values),final));forwards+=1
   if forward_error==forwards:raise injected
  def vote():
   nonlocal votes
   votes+=1;value=votes==cancel_vote;events.append(('vote',value));return value
  def discard():events.append(('discard',))
  def sample():
   events.append(('sample',))
   if discard_inside_sample:discard()
   return 7
  try:
   kwargs=dict(prefill=prefill,window=window,forward=forward,vote=vote,sample=sample,discard=discard)
   result=observed(plan,0,record=record,executor=execute,**kwargs)if traced else execute(plan,0,**kwargs)
   error=None
  except RuntimeError as err:
   assert err is injected;result=None;error=type(err).__name__
  return events,result,error,record
 cases=[]
 for n in[1,2,3,4,5]:
  for w in[2,4]:
   baseline=run(n,w,traced=False,discard_inside_sample=True);candidate=run(n,w,traced=True,discard_inside_sample=True)
   assert baseline[:3]==candidate[:3];a=candidate[3][0];expected=[min(w,n-start)for start in range(0,n,w)]
   assert a['actual_committed_group_chunks']==expected and a['successful_forward_count']==n and a['execute_discard_calls']==1 and a['vote_callbacks']==len(expected)+1 and a['sample_calls']==1 and a['complete_prefill_window_counter_qualified']
   assert a['W4_group_larger_than_W2_observed']==(w==4 and n>2)
   cases.append(dict(kind='normal',chunks=n,window=w,actual_groups=expected,existing_callback_sequence_identical=True))
 for w in[2,4]:
  for cancel in[1,2]:
   a=run(5,w,traced=False,cancel_vote=cancel);b=run(5,w,traced=True,cancel_vote=cancel);assert a[:3]==b[:3];trace=b[3][0]
   assert trace['completed']and trace['canceled']and not trace['complete_prefill_window_counter_qualified']and trace['sample_calls']==0 and trace['execute_discard_calls']==1
   cases.append(dict(kind='cooperative_cancel_or_disconnect_vote',window=w,vote=cancel,existing_callback_sequence_identical=True))
 for w in[2,4]:
  a=run(5,w,traced=False,forward_error=2);b=run(5,w,traced=True,forward_error=2);assert a[:3]==b[:3];trace=b[3][0]
  assert trace['error']=='RuntimeError'and not trace['completed']and len(trace['forward_calls'])==2 and trace['successful_forward_count']==1 and not trace['forward_calls'][-1]['returned']and trace['execute_discard_calls']==1
  cases.append(dict(kind='Native_error_keeps_primary_and_partial_counter',window=w,existing_callback_sequence_identical=True))
 a=run(1,1,traced=False,prefill=False);b=run(1,1,traced=True,prefill=False);assert a[:3]==b[:3]and not b[3];cases.append(dict(kind='decode_delegates_unwrapped',existing_callback_sequence_identical=True))
 # Reverse the four exact nonmath source edits, then compare the whole owner text.
 parent=(P/'native_owner.py').read_text();body=(D/'native_owner_audited.py').read_text()
 body=body.replace('from .glm_prefill_execute_r236 import execute_plan\nfrom .prefill_window_audit import execute_observed','from .glm_prefill_execute_r236 import execute_plan')
 body=body.replace("report['attempted_Native']=request_index;report['active_request_calls']=[];report['active_request_prefill_window_audit']=[];prefill_audit=report['active_request_prefill_window_audit'];save()","report['attempted_Native']=request_index;report['active_request_calls']=[];save()")
 body=body.replace('past,token,canceled=execute_observed(plan,past,prefill=step==0,window=prefill_window if step==0 else 1,forward=forward,vote=vote,sample=sample,discard=discard,record=prefill_audit,executor=execute_plan)','past,token,canceled=execute_plan(plan,past,prefill=step==0,window=prefill_window if step==0 else 1,forward=forward,vote=vote,sample=sample,discard=discard)')
 body=body.replace('row.update(prefill_window_audit=copy.deepcopy(prefill_audit),admitted_index=pair_index','row.update(admitted_index=pair_index')
 assert body==parent
 assert all(n not in sys.modules for n in ['torch','transformers','infinilm','_infinilm'])
 return dict(passed=True,cases=cases,case_count=len(cases),delegate_SHA256=sha(P/'glm_prefill_execute_r236.py'),observer_SHA256=sha(D/'prefill_window_audit.py'),parent_owner_SHA256=sha(P/'native_owner.py'),audited_owner_SHA256=sha(D/'native_owner_audited.py'),original_complete_owner_except_four_trace_edits_identical=True,Native_math_or_call_schedule_changed=False,Native_dlopen=False,actualcheckpoint_encoding=False,recording_leaves=['forward','vote cancellation/control','sample','discard'],Slurm=0,GPU=0,Model=0,scope='Actual original executor with recording callbacks and reversible private owner source handoff; no device quality/exposure qualification')
if __name__=='__main__':print(json.dumps(check(),indent=2))
