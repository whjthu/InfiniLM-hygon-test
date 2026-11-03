"""Changed deployment consumers only. HTTP/Native/Slurm leaves are recording.

Actual public CLI, complete original fixture bytes, two-transport client parser,
Run STOP, terminal collector and controller.main are consumed; no device calls.
"""
from pathlib import Path
import contextlib,copy,hashlib,io,json,os,sys,tempfile,time,types
from unittest.mock import patch
D=Path(__file__).resolve().parent;G=D.parents[2];P=G/'code/bw1000/public-GLM-native-provider-r425'
sys.path[:0]=[str(D),str(P)]
import client,collect,controller,preflight
from public_glm_native.protocol import Run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
F=G/'code/bw1000/GLM-W4-cross-workload-r349/fixtures'

def fixtures(temp):
 card=json.loads((D/'FIXTURE_CARD.json').read_bytes())
 for f in card['fixtures']:f['request_external_path']=str(F/f['local_untracked_request_file'])
 path=temp/'fixtures.json';path.write_text(json.dumps(card));return card,path

def reports(folder,identity,card):
 folder.mkdir(parents=True,exist_ok=True)
 def write(name,q):
  (folder/name).write_text(json.dumps(q))
 write('MODEL_READY.json',dict(identity=identity,Model_READY=True,actual_owner_count=16,
  all16_loaded_postload4GiB_and_service_agreement=True,prepared_identity=dict(eos_token_ids=[999])))
 write('FRONTEND.json',dict(identity=identity,PID=1001))
 for stage in range(2):
  write('leader-%d.json'%stage,dict(PID=1000+stage));write('actors-%d.json'%stage,dict(owned_process_PIDs=list(range(2000+stage*8,2008+stage*8))))
  for rank in range(8):
   owners=[dict(device=i,changes=1,verified_mask=','.join(map(str,range(i*4,i*4+4))))for i in range(8)]
   aff=dict(width=4,actual_owner_report=dict(owners=owners));rows=[]
   for index,f in enumerate(card['fixtures']):
    call=dict(CAPI_returned=True,CAPI_RC=0,finite=True if stage==1 else None,Native_CAPI_s=.1)
    rows.append(dict(request_id='new-request-'+str(index),native_profile=card['plan'][index]['profile'],
     actual_prompt_token_ids_SHA256=f['current_prepared_prompt_IDs_SHA256'],prompt_tokens=f['current_prepared_prompt_tokens'],
     requested_output_budget=4096,generated_natural_EOS=True,native_finish_reason='eos_token',reason='eos_token',
     generated_token_ids=[10,11,999],generated_token_ids_SHA256='recording-equal-'+str(index),generated_tokens=3,
     normal_reset_before=True,normal_reset_after=True,original_reset_all16_completed=True,
     final_metadata=dict(original_reset_all16_completed=True),native_calls=[call],backend_token_monotonic=[2.,3.,4.],started_monotonic=1.))
   q=dict(actual_step=identity['actual_step'],source_manifest_SHA256=identity['source_manifest_SHA256'],completed=True,
    normal_shutdown=True,worker_shutdown_completed=True,post_close_Torch_liveness=True,
    all16_preweight_8GiB_passed=True,all16_loaded_and_postload_4GiB_passed=True,
    owner_affinity_after_load=aff,owner_affinity_final=copy.deepcopy(aff),attempted_Native=2,completed_Native=2,service_requests=rows)
   # Nonmask diagnostic counters may legitimately differ; strict masks remain.
   q['owner_affinity_final']['actual_owner_report']['owners'][rank]['diagnostic_calls']=9
   write('owner-%d-%d.json'%(stage,rank),q)
   write('lifecycle-%d-%d.json'%(stage,rank),dict(normal=True,destroy_attempts=1,shutdown_attempts=1,destroy_RC=0,shutdown_RC=0))

class Response:
 def __init__(self,raw,stream):self.raw=raw;self.status=200;self.headers={'Content-Type':'text/event-stream'if stream else'application/json'};self.position=0
 def __enter__(self):return self
 def __exit__(self,*args):return False
 def read(self,n=-1):
  value=self.raw[self.position:]if n<0 else self.raw[self.position:self.position+n];self.position+=len(value);return value
 def __iter__(self):return iter(self.raw.splitlines(keepends=True))

def run_client(root,card,fixturefile,identity,*,bad=False):
 folder=root/'results/public-GLM-provider-r425'/controller.RUN;reports(folder,identity,card);sent=[]
 def open_http(request,timeout):
  body=json.loads(request.data);index=len(sent);sent.append(request);f=card['fixtures'][index]
  assert body['stream']==bool(index)and request.get_header('X-infinilm-native-profile')==card['plan'][index]['profile']
  _,original,gold=client.body_and_gold(f);assert body['messages']==original['messages']and body['max_tokens']==4096
  if not index:assert request.data==(F/f['local_untracked_request_file']).read_bytes()
  args=copy.deepcopy(gold)
  if 'content'in args:args['content']=json.dumps(args['content'])
  if bad:args['a']=-1
  tool=dict(id='new-call-'+str(index),type='function',function=dict(name=f['gold']['tool'],arguments=json.dumps(args)))
  usage=dict(prompt_tokens=f['current_prepared_prompt_tokens'],completion_tokens=3,total_tokens=f['current_prepared_prompt_tokens']+3)
  wire=dict(id='new-request-'+str(index),choices=[dict(index=0,message=dict(role='assistant',tool_calls=[tool]),finish_reason='tool_calls')],usage=usage)
  if index:
   raw=('data: '+json.dumps(dict(id=wire['id'],choices=[dict(index=0,delta=dict(tool_calls=[dict(index=0,**tool)]),finish_reason='tool_calls')],usage=usage))+'\n\ndata: [DONE]\n\n').encode()
  else:raw=json.dumps(wire).encode()
  return Response(raw,bool(index))
 out=root/'client-output'
 args=['--base-url','http://127.0.0.1:43843/v1','--runtime-dir',str(root/'results/public-GLM-provider-r425'),
  '--run',controller.RUN,'--expected-step',identity['actual_step'],'--source-sha',identity['source_manifest_SHA256'],
  '--config-sha',identity['provider_config_SHA256'],'--fixtures',str(fixturefile),'--output-dir',str(out),'--deadline-unix',str(time.time()+500)]
 with patch.object(client.urllib.request,'urlopen',open_http),patch.object(client,'wait_listener',lambda *args:None):rc=client.main(args)
 q=json.loads((out/'result.json').read_bytes());assert q['completed']and q['passed']==(not bad)and rc==(1 if bad else 0)
 assert q['attempted_HTTP']==(1 if bad else 2)and q['completed_HTTP']==(0 if bad else 2)
 return out/'result.json'

def main():
 cases=[];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 actual=preflight.check(D,P,m,fixture_root=F);cases.append(dict(name='actual_final_CLI_both_roles_and_complete_assets',passed=True,result=actual))
 with tempfile.TemporaryDirectory()as temp:
  root=Path(temp);card,ff=fixtures(root);identity=dict(run=controller.RUN,actual_step='910376.99999',source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=time.time(),Native_seconds=5100)
  output=run_client(root,card,ff,identity)
  summary=collect.collect(root,controller.RUN,identity['actual_step'],identity['source_manifest_SHA256'],output,time.time()+5)
  assert summary['actual_owner_count']==16 and summary['Native_CAPI_calls']==32
  cases.append(dict(name='actual_HTTP_SSE_client_and_all16_terminal_collector',passed=True,recording_leaves=['HTTP responses','Native reports','listener'],external_requests=2))
 with tempfile.TemporaryDirectory()as temp:
  root=Path(temp);card,ff=fixtures(root);run_client(root,card,ff,identity,bad=True)
  cases.append(dict(name='bad_integer_gold_first_request_persist_then_stop',passed=True))
 # Real controller.main: subprocess/clock/device leaves only are recording.
 with tempfile.TemporaryDirectory()as temp:
  root=Path(temp);(root/'results').mkdir();(root/'logs').mkdir();card,ff=fixtures(root);processes={};counter=[0];source=D
  def fake_popen(cmd,**kw):
   assert kw['start_new_session']and kw['stdin']==controller.subprocess.DEVNULL
   label=next(v.split('=',1)[1]for v in cmd if v.startswith('--job-name='));counter[0]+=1
   class Proc:
    pid=10000+counter[0]
    def poll(self):return 0 if label.endswith('-client')or(root/'results/public-GLM-provider-r425'/controller.RUN/'STOP.json').exists()else None
   processes[label]=Proc()
   if not label.endswith('-client'):
    epoch=float(kw['env']['GLM_R425_GPUSTART']);ident=dict(identity,gpu_start_epoch=epoch)
    reports(root/'results/public-GLM-provider-r425'/controller.RUN,ident,card)
   else:
    ident=json.loads((root/'results/public-GLM-provider-r425'/controller.RUN/'MODEL_READY.json').read_bytes())['identity']
    original=run_client(root,card,ff,ident);out=root/'results'/(controller.LABEL+'-'+controller.RUN+'-client');out.mkdir();(out/'result.json').write_bytes(original.read_bytes())
   return processes[label]
  def output(cmd,**kw):
   if cmd[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=cpuuser(123) NodeList=a02r4n[12-13]'
   if cmd[:3]==['scontrol','show','hostnames']:return '\n'.join(controller.NODES)
   assert cmd[0]=='sacct';values=[]
   for label,proc in processes.items():values.append('|'.join(['910376.'+('99998'if label.endswith('-client')else'99999'),label,'COMPLETED'if proc.poll()==0 else'RUNNING','0:0','a02r4n[12-13]']))
   return '\n'.join(values)
  def idle(cmd,**kw):
   assert '--gres=none'in cmd and cmd[-1].endswith('idle.py')
   return types.SimpleNamespace(returncode=0,stderr='',stdout='\n'.join(json.dumps(dict(node=n,CPU_only=True,GPU_initialized=False,kfd_pids=[],matching_component_processes=[],expected_PIDs_still_exist=[],ports_bindable={'43843':True},physical_PCI=list(map(str,range(8)))))for n in controller.NODES))
  proof=lambda *args:actual
  with patch.multiple(controller,R=root,D=source,PACKAGE=P),patch.object(controller,'verify',lambda digest:m),patch.object(preflight,'check',proof),patch.dict(os.environ,dict(SLURM_JOB_ID='910407',SLURM_STEP_ID='88888',USER='cpuuser')),patch.object(controller.socket,'gethostname',lambda:'a14r2n20'),patch.object(controller.subprocess,'Popen',fake_popen),patch.object(controller.subprocess,'check_output',output),patch.object(controller.subprocess,'run',idle),patch.object(controller.time,'sleep',lambda seconds:None),contextlib.redirect_stdout(io.StringIO()):
   rc=controller.main(['--manifest-sha','recording','--publication-commit','0'*40])
  result=json.loads((root/'results'/(controller.LABEL+'-'+controller.RUN+'-watch.json')).read_bytes())
  assert rc==0 and result['passed']and result['normal_STOP_published']and not result['forced_cleanup']
  assert len(processes)==2 and result['actual_GPU']=='910376.99999'and result['actual_client']=='910376.99998'
  cases.append(dict(name='actual_controller_main_single_submission_normal_STOP_and_post',passed=True,recording_leaves=['Slurm/Popen','device/filesystem Native evidence','passive idle']))
 q=dict(completed=True,passed=True,cases=cases,old_leaf_suites_rerun=False,GPU=0,Native_dlopen=0,Model_READY=False)
 (D/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
