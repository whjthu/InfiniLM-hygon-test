"""Changed final public deployment consumers, explicit process/device/wire leaves."""
from pathlib import Path
import contextlib,copy,hashlib,io,json,os,sys,tempfile,time,types
from unittest.mock import patch
D=Path(__file__).resolve().parent;G=D.parents[3];P=G/'code/bw1000/public-Deep-native-provider-r427/source-v3';V=D.parent/'source-v1';L=G.parent
sys.path[:0]=[str(D),str(P)]
import client,client_launch,collect,controller,preflight
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures(folder):
 card=json.loads((D/'FIXTURE_CARD.json').read_bytes());card['fixture_path']=str(L/card['local_untracked_fixture']);p=folder/'fixtures.json';p.write_text(json.dumps(card));return card,p
def reports(folder,identity):
 folder.mkdir(parents=True,exist_ok=True)
 def write(n,q):(folder/n).write_text(json.dumps(q))
 write('FRONTEND.json',dict(identity=identity,PID=1001))
 write('MODEL_READY.json',dict(identity=identity,Model_READY=True,Native_internal_owner_count=8,HC_mode=0,TP=8,PP=1))
 write('closed-engine.json',dict(identity=identity,normal=True,engine_destroy_returned_zero=True,observed_all8_threads_exited=True))
 rows=[]
 for i,arm in enumerate(json.loads((D/'FIXTURE_CARD.json').read_bytes())['plan']):
  tokens=8489;h=dict(qualified=True,prequeue_bound=True,same_prepared_encoder_reproduces_prompt_and_IDs=True,markers_sent_to_Native=False,content_one_contribution_each_and_ordered=True,all_content_contribution_slices_equal=True,output_budget=arm['budget'],prompt_tokens=tokens,Native_prompt_token_ids_SHA256='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df',pending_thinking_prefix_open=arm['thinking'])
  rows.append(dict(request_id='request-'+str(i),admission=dict(history_render_audit=h,text_delivery=arm['delivery']),prompt_tokens=tokens,max_new_tokens=arm['budget'],native_finish_reason='eos_token',normal_reset_before=True,normal_reset_after=True,generated_token_ids=[10,11,1],actual_generated_tokens=3,actual_forward_attempts=11,actual_forward_completed=11,Native_continuation=[dict(kind='prefill',past=s,tokens=min(1024,tokens-s),total=min(s+1024,tokens))for s in range(0,tokens,1024)]+[dict(kind='decode_summary',calls=2,first_past=8489,last_total=8491)],r449_index=i,r449_mode=arm['mode'],r449_warm=arm['warmup'],r449_quiescent=True,r449_poisoned=False,r449_r366_fixed=0,r449_actual_T1_coverage=dict(passed=True,HC_groups=86,completed_forwards=11,one_token_prefill_forwards=0,decode_forwards=2,T1_HC_per_owner=172,mode=arm['mode'],r366_fixed=0),r449_actual_owner_counts=[dict(owner=owner,created=86,ready=86,legacy=dict(created=0,ready=0,pre=946,eligible_bf16_t1=172,completed=0,other_t=774,f32_t1=0),fullpre=dict(observed=946*arm['mode'],eligible=172*arm['mode'],selected=172*arm['mode'],completed=172*arm['mode'],prefill=774*arm['mode'],f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0))for owner in range(8)],backend_token_monotonic=[2.,3.,4.],started_monotonic=1.,Native_forward_host_summary=dict(prefill_calls=9,prefill_s=.1,decode_calls=2,decode_s=.2),final_metadata=dict(recording=True,buffered_P0_text=arm['delivery']=='P0')))
 owners=[dict(device=i,changes=1,verified_mask=','.join(map(str,range(i*4,i*4+4))))for i in range(8)]
 write('owner.json',dict(identity=identity,completed=True,normal_Native_shutdown=True,engine_worker_shutdown_completed=True,observed_all8_threads_exited=True,loaded_key_count=67569,Native_SHA256='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1',HC_mode=0,owner_affinity_after_weight_load=dict(actual_owner_report=dict(owners=owners)),owner_thread_observation=dict(status='OBSERVED',owners=[dict(TID=2000+i)for i in range(8)]),requests=rows,HC_dispatch_checked_forwards=66,layout_dispatch_checked_forwards=66))
class Response:
 def __init__(self,raw,stream):self.raw=raw;self.status=200;self.headers={'Content-Type':'text/event-stream'if stream else'application/json'};self.position=0
 def __enter__(self):return self
 def __exit__(self,*a):return False
 def read(self,n=-1):
  value=self.raw[self.position:]if n<0 else self.raw[self.position:self.position+n];self.position+=len(value);return value
 def __iter__(self):return iter(self.raw.splitlines(keepends=True))
def run_client(root,card,ff,identity,*,bad=False,bad_hits=False,args=None):
 folder=root/'results/Deep-HC-fullpre-public-r449'/controller.RUN
 if not folder.exists():reports(folder,identity)
 if bad_hits:
  p=folder/'owner.json';report=json.loads(p.read_bytes());report['requests'][1]['r449_actual_owner_counts'][7]['fullpre']['completed']=0;p.write_text(json.dumps(report))
 sent=[]
 def open_http(request,timeout):
  body=json.loads(request.data);i=len(sent);sent.append(request);arm=card['plan'][i]
  raw,original,gold,h=client.body_and_gold(card,arm);assert body==original and request.data==raw
  if bad:gold['total']+=1
  content='```json\n'+json.dumps(gold)+'\n```';usage=dict(prompt_tokens=8489,completion_tokens=3,total_tokens=8489+3)
  message=dict(role='assistant',content=content)
  message['reasoning_content']='recording reasoning; actual parser route consumer'
  wire=dict(id='request-'+str(i),choices=[dict(index=0,message=message,finish_reason='stop')],usage=usage)
  if arm['stream']:
   raw=('data: '+json.dumps(dict(id=wire['id'],choices=[dict(index=0,delta=message,finish_reason='stop')],usage=usage))+'\n\ndata: [DONE]\n\n').encode()
  else:raw=json.dumps(wire).encode()
  return Response(raw,arm['stream'])
 out=root/'client-output'
 if args is None:args=['--base-url','http://127.0.0.1:43882/v1','--runtime-dir',str(root/'results/Deep-HC-fullpre-public-r449'),'--run',controller.RUN,'--expected-step',identity['actual_step'],'--source-sha',identity['source_manifest_SHA256'],'--config-sha',identity['provider_config_SHA256'],'--fixtures',str(ff),'--output-dir',str(out),'--deadline-unix',str(time.time()+500)]
 else:out=Path(args[args.index('--output-dir')+1])
 with patch.object(client.urllib.request,'urlopen',open_http),patch.object(client,'wait_listener',lambda *args:None):rc=client.main(args)
 q=json.loads((out/'result.json').read_bytes());failed=bad or bad_hits;assert q['completed']and q['passed']==(not failed)and rc==(1 if failed else 0),q
 assert q['attempted_HTTP']==(1 if bad else 2 if bad_hits else 6)and q['completed_HTTP']==(0 if bad else 1 if bad_hits else 6)
 if not failed:
  assert [x['delivery']for x in q['rows']]==['P1']*6 and [x['mode']for x in q['rows']]==[0,1,0,1,1,0]
  assert all(x['client_clock']['first_reasoning_payload_s']is not None and x['client_clock']['first_content_payload_s']is not None and x['client_clock']['client_per_token_speed'].startswith('UNKNOWN')for x in q['rows'])
 return out/'result.json'
def main():
 cases=[];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());actual=preflight.check(D,V,m,fixture_root=L,evidence_root=L,cli_leaf=lambda d,m:dict(source_and_schema_valid=True,recording_container_CLI=True))
 cases.append(dict(name='actual_CPU750_938_evidence_scopes_and_frozen010110_same_fixture_plan',passed=True,proof=actual))
 identity=dict(run=controller.RUN,actual_step='910374.99999',source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=time.time(),Native_seconds=2580)
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);card,ff=fixtures(root);out=run_client(root,card,ff,identity)
  from public_deep_native.protocol import Run
  Run(dict(runtime_directory=str(root/'results/Deep-HC-fullpre-public-r449'),**identity)).stop()
  result=collect.collect(root,controller.RUN,identity['actual_step'],identity['source_manifest_SHA256'],out,time.time()+5)
  assert result['actual_owner_count']==8 and result['Native_CAPI_calls']==66 and result['external_requests']==6
  cases.append(dict(name='actual_six_SSE_client_mode_ownercoverage_clocks_gold_and_collector',passed=True,recording_leaves=['HTTP wire','Native reports','listener']))
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);card,ff=fixtures(root);run_client(root,card,ff,identity,bad=True)
  cases.append(dict(name='wrong_integer_gold_first_request_persist_FAIL',passed=True))
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);card,ff=fixtures(root);run_client(root,card,ff,identity,bad_hits=True)
  cases.append(dict(name='actual_client_first_ON_missing_owner_hit_stops_after_two_HTTP',passed=True))
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'results').mkdir();(root/'logs').mkdir();source=root/'deploy';source.mkdir()
  for p in D.iterdir():
   if p.is_file():(source/p.name).write_bytes(p.read_bytes())
  card,ff=fixtures(root);(source/'FIXTURE_CARD.json').write_bytes(ff.read_bytes());processes={};counter=[0]
  def fake_popen(cmd,**kw):
   assert kw['start_new_session']and kw['stdin']==controller.subprocess.DEVNULL
   label=next(v.split('=',1)[1]for v in cmd if v.startswith('--job-name='));counter[0]+=1
   class Proc:
    pid=10000+counter[0]
    def poll(self):return 0 if label.endswith('-client')or(root/'results/Deep-HC-fullpre-public-r449'/controller.RUN/'STOP.json').exists()else None
   processes[label]=Proc()
   if not label.endswith('-client'):
    assert '-c32'in cmd and'--mem=128000M'in cmd and'00:45:00'in cmd
    assert kw['env']['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'
    ident=dict(identity,gpu_start_epoch=float(kw['env']['DEEP_R449_MODEL_GPUSTART']));reports(root/'results/Deep-HC-fullpre-public-r449'/controller.RUN,ident)
   else:
    i=next(i for i,v in enumerate(cmd)if Path(v).name=='client_launch.py');argv=cmd[i:]
    def execute(path,values):
     ident=json.loads((root/'results/Deep-HC-fullpre-public-r449'/controller.RUN/'MODEL_READY.json').read_bytes())['identity']
     run_client(root,card,source/'FIXTURE_CARD.json',ident,args=values[4:])
    with patch.object(sys,'argv',argv),patch.object(client_launch.os,'execv',execute),patch.dict(os.environ,dict(SLURM_JOB_ID='910374',SLURM_STEP_ID='99998')):client_launch.main()
   return processes[label]
  def output(cmd,**kw):
   if cmd[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=cpuuser(123) NodeList=a02r4n04'
   if cmd[:3]==['scontrol','show','hostnames']:return 'a02r4n04'
   assert cmd[0]=='sacct';return '\n'.join('|'.join(['910374.'+('99998'if n.endswith('-client')else'99999'),n,'COMPLETED'if p.poll()==0 else'RUNNING','0:0','a02r4n04'])for n,p in processes.items())
  def idle(cmd,**kw):
   assert '--gres=none'in cmd and cmd[-1].endswith('idle.py')
   return types.SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(node='a02r4n04',CPU_only=True,GPU_initialized=False,kfd_pids=[],matching_component_processes=[],expected_PIDs_still_exist=[],ports_bindable={'43882':True},physical_PCI=list(map(str,range(8))))))
  with patch.multiple(controller,R=root,D=source,PACKAGE=P),patch.object(controller,'verify',lambda digest:m),patch.object(preflight,'check',lambda *args:actual),patch.dict(os.environ,dict(SLURM_JOB_ID='910407',SLURM_STEP_ID='88888',USER='cpuuser')),patch.object(controller.socket,'gethostname',lambda:'a14r2n20'),patch.object(controller.subprocess,'Popen',fake_popen),patch.object(controller.subprocess,'check_output',output),patch.object(controller.subprocess,'run',idle),patch.object(controller.time,'sleep',lambda s:None),contextlib.redirect_stdout(io.StringIO()):rc=controller.main(['--manifest-sha','recording','--publication-commit','0'*40])
  result=json.loads((root/'results'/(controller.LABEL+'-'+controller.RUN+'-watch.json')).read_bytes())
  assert rc==0 and result['passed']and result['normal_STOP_published']and not result['forced_cleanup'],result
  assert len(processes)==2 and result['actual_GPU']=='910374.99999'and result['actual_client']=='910374.99998'
  cases.append(dict(name='actual_controller_main_GPU_client_argv_identity_STOP_post',passed=True,recording_leaves=['Slurm/Popen','Native reports','passive idle','HTTP wire'],actual_client_launch_consumed=True))

 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);folder=root/'reports';reports(folder,identity);data=json.loads((folder/'owner.json').read_bytes())
  from model_counts import validate
  validate(data['requests'][1],1)
  for label,change in [('wrong_mode',lambda r:r.update(r449_mode=0)),('missing_owner_hit',lambda r:r['r449_actual_owner_counts'][7]['fullpre'].update(completed=0)),('unfinished_forward',lambda r:r.update(actual_forward_completed=10))]:
   row=copy.deepcopy(data['requests'][1]);change(row)
   try:validate(row,1)
   except (AssertionError,ValueError,KeyError):pass
   else:raise AssertionError('Missing rejection:'+label)
  cases.append(dict(name='production_validator_rejects_wrong_mode_owner_hit_and_incomplete_forward',passed=True,recording_only=True))
 q=dict(completed=True,passed=True,cases=cases,old_leaf_suites_rerun=False,nine_semantics_not_replayed=True,whole_container_CLI_pending_actual_preflight=True,GPU=0,Native_dlopen=0,Model_READY=False)
 (D/'LOCAL_DEPLOYMENT_CPU.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
