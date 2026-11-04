"""Only new session/audit/grade filehandoff; recording leaves, no old suites."""
import ctypes as C,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from fixed_cohort import native_context as owner
from fixed_cohort.profile import config
from fixed_cohort.recording import Recording,RecordingTorch
import session
import precision

class OriginalMathRecording(Recording):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.policy_values=[]
  def setter(value):self.policy_values.append(value);return 0 if value==0 else 1
  cb=C.CFUNCTYPE(C.c_int,C.c_int)(setter);self.keep.append(cb);self.lib.modern_deep_precision_policy_r3443=cb
 def leaf(self,name,*args):
  before=self.fp32;rc=super().leaf(name,*args)
  if name=='modern_native_engine_forward':self.fp32=before
  return rc

def run(prepared=None):
 r=OriginalMathRecording();report={};saves=[]
 if prepared is None:prepared=dict(config=config('recording-only'),receipt={'library':'recording-native-no-dlopen'},mapping={},
  cfg={'vocab_size':129280,'num_hidden_layers':43,'compress_ratios':[4]*43},
  records=[dict(row=i,case_id='recording-'+str(i),prompt_token_ids=tuple(range(1050+i)),max_new_tokens=4096,
     gold=dict(suite='gsm8k'if i<2 else'mmlu',gold_numeric='12345',gold_choice='D'))for i in range(4)])
 def weight_leaf(binding,ctx,**kwargs):
  assert ctx==r.ctx;kwargs['report']['loaded_key_count']=67569
 affinity={'actual_owner_report':{'owners':[]},'recording_only':True}
 policy=dict(failures=0,owners=[dict(tid=i,caller='libinfiniops.so',created=1,verified=1)for i in range(8)])
 class Decoder:
  def decode(self,*a,**kw):return 'INVALID NO ANSWER' # deliberately parsefailed; must still run24
 with tempfile.TemporaryDirectory()as directory:
  with patch.object(owner,'providers',lambda *a:{'recording_only':True}),patch.object(owner,'verify_affinity',lambda *a:affinity),patch.object(owner,'load_checkpoint',weight_leaf),patch.object(owner,'observe_threads',lambda *a:{'status':'RECORDING','owners':[]}),patch.object(owner,'blas_snapshot',lambda:policy),patch.object(session,'owner_affinity',lambda *a:affinity),patch.object(precision,'verify_provider',lambda *a:{'recording_only':True}):
   session.run_session(prepared,dict(gpu_start_epoch=time.time()),binding=r.binding,torch=RecordingTorch(),safe_open=None,tokenizer=Decoder(),output=directory,report=report,save=lambda:saves.append(report.get('phase')),deadline_check=lambda *a,**kw:None)
  assert report['completed_cohorts']==6 and report['logical_requests']==24
  assert len(report['fixed_cohort_trials'])==6 and report['all_cohorts_completed_normal']
  assert [v['mode']for v in report['fixed_cohort_trials']]==['serial','cohort','serial','cohort','cohort','serial']
  assert all(not q['score']['correct']for v in report['fixed_cohort_trials']for q in v['quality'])
  assert r.destroy_calls==1 and r.reset_calls==12 and not r.callback_errors
  assert r.policy_values==[0]and r.fp32==0
  assert sum(v['kind']=='create'for v in r.calls)==1
  lines=[json.loads(s)for s in (Path(directory)/'calls-and-events-private.jsonl').read_text().splitlines()]
  calls=[v for v in lines if v['event']=='CAPI_return'];assert len(calls)==report['attempted_CAPI']==report['completed_CAPI']
  assert {v['count']for v in calls}=={1,2,3,4}
  assert all(len(v['all8_counters'])==8 and v['input_readonly']and v['BF16_nonfinite_count']==0 for v in calls)
  assert len(list(Path(directory).glob('cohort-*-private.json')))==6
  return dict(passed=True,actual_new_session=True,actual_scoped_CAPI_audit=True,actual_parser_and_scorer=True,actual_glue_filehandoff=True,
    one_Context_create=1,normal_destroy_recording=1,sync_resetall_recording=12,cohorts=6,logical_requests=24,
    all24_wrong_not_early_stop=True,CAPI=len(calls),original_precision_policy0=True,all8_FP32_delta0_recorded=True,actual_Native=False,Native_leaves_recorded=True,GPU=False,Model_READY=False,
    old967_cases_or945_encoding_replayed=False)

if __name__=='__main__':print(json.dumps(run(),indent=2))
