"""Actual Native.call/strictverify with ctypes recording CABI and real files only."""
from pathlib import Path
import ast,json,os,tempfile,types
from ordinary import Native
from native_binding import declare
from model_affinity_check_r2628 import verify
from check_entry_CPU import RecordingLibrary
D=Path(__file__).resolve().parent
class Control:
 def collect(self,label,p):
  if 'status'in p:return[dict(node=i,status=p['status'])for i in range(4)]
  return[dict(p,node=i,flags=3 if i==3 else 0,finite=p['finite']if i==3 else None,selected=p['selected']if i==3 else None)for i in range(4)]
class Library(RecordingLibrary):
 def dispatch(self,name,*a):
  if name=='modern_owner_affinity_report_r2392'and self.forward_calls and self.mode in['never_bound','overlapping_mask']:
   return json.dumps(dict(owners=[dict(device=d,changes=0 if self.mode=='never_bound'else 1,verified_mask=''if self.mode=='never_bound'else '0,1,2,3')for d in range(8)])).encode()
  return super().dispatch(name,*a)
def case(mode):
 mapping={'language_model.model.layers.'+str(i)+'.dummy':'recording-only'for i in range(93)};mapping['language_model.lm_head.weight']='recording-only'
 lib=declare(Library(3,mapping,{'retained8635':[42,163585]},mode=mode));lib.reset_count=1
 initial=json.loads(lib.modern_owner_affinity_report_r2392());assert all(v['changes']==0 and not v['verified_mask']for v in initial['owners'])
 try:verify(lib);raise RuntimeError('Premature check unexpectedly passed')
 except AssertionError:pass
 with tempfile.TemporaryDirectory(prefix='Kimi410-affinity-phase-')as tmp:
  output=Path(tmp)/'actual-call.json';report={};saved=[]
  def save():
   output.write_text(json.dumps(report,allow_nan=False)+'\n');saved.append(json.loads(output.read_bytes()))
  gate=types.SimpleNamespace(check=lambda label:None)
  native=Native(lib,lib.context,Control(),3,gate,report,save)
  try:
   native.call(0,0,[42])
   assert mode=='normal';assert report['owner_affinity_after_first_forward']['verified']and len(lib.forward_calls)==1
   native.call(1,1,[42]);assert len(lib.forward_calls)==2 and len(report['ordinary_calls'])==2
   final=verify(lib);assert final['verified']and all(v['changes']==1 for v in final['actual_owner_report']['owners'])
  except AssertionError:
   assert mode!='normal';assert len(lib.forward_calls)==1 and len(report['ordinary_calls'])==1
   persisted=json.loads(output.read_bytes());assert persisted['ordinary_calls'][0]['CAPI_return']==0 and not persisted.get('owner_affinity_after_first_forward')
  assert saved[0]['ordinary_calls'][0]['CAPI_return']==0 and 'owner_affinity_after_first_forward'not in saved[0]
  return dict(case=mode,passed=True,prework_changes=0,recording_CAPI_calls=len(lib.forward_calls),first_completed_call_persisted_before_verify=True,strict_verify_actual_Python=True,Native_dlopen=False,real_CPU_affinity_binding=False)
def main():
 os.environ['INFINILM_OWNER_CPU_WIDTH_R2392']='4'
 cases=[case(x)for x in['normal','never_bound','overlapping_mask']]
 s=(D/'bootstrap.py').read_text();a=s.index("report['owner_affinity_before_weight_load']");b=s.index('  save()',a);assert '.verify('not in s[a:b]and 'os.sched_getaffinity(0)'in s[a:b]
 assert "report['owner_affinity']=deps.modules['model_affinity_check_r2628'].verify(lib)"in s
 print(json.dumps(dict(completed=True,passed=True,cases=cases,GPU=0,Slurm=0,Native_dlopen=False,Runtime_created=False,Model=0,source='Shared production Native.call and original strictverify; CABI leaf records lazily on successfulforward, real filesystem save, not actual owner OS-affinity qualification')))
if __name__=='__main__':main()
