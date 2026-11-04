"""Only new two-case deployment handoff; HTTP/Slurm/Native are recording leaves."""
from pathlib import Path
import contextlib,copy,hashlib,io,json,os,sys,tempfile,time,types
from unittest.mock import patch
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import client,client_launch,collect,controller,preflight,fixture
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy_private(folder):
 # Exact existing artifacts, no encoding or Native calls. Local development
 # reads the sealed450 archive; remote CPU reads its original945 files.
 if os.environ.get('DEEP457_LOCAL_INPUTS'):
  base=Path(os.environ['DEEP457_LOCAL_INPUTS'])
  for n in ['cases.json','requests-private.json','encoding.json','CPU962.json']:(folder/n).write_bytes((base/n).read_bytes())
 else:
  card=json.loads((D/'FIXTURE_CARD.json').read_bytes())
  for key,name in [('cases','cases.json'),('requests','requests-private.json'),('encoding','encoding.json')]:
   path=Path(card[key+'_path']);(folder/name).write_bytes(path.read_bytes())
  pin=json.loads((D/'CPU_EVIDENCE.json').read_bytes());(folder/'CPU962.json').write_bytes(Path(pin['CPU962']['path']).read_bytes())
 return folder
def reports(folder,identity,card):
 folder.mkdir(parents=True,exist_ok=True)
 def write(n,q):(folder/n).write_text(json.dumps(q))
 fixed=dict(profile='deep-hc0-fullpre-c1',fixed_mode=1,Native_mode_readback=1,HC_mode=0,r366_fixed=0,C1=True,HC_groups=86,provider={'recording':True})
 write('FRONTEND.json',dict(identity=identity,PID=1001));write('MODEL_READY.json',dict(identity=identity,Model_READY=True,Native_internal_owner_count=8,HC_mode=0,TP=8,PP=1,fixed_profile=fixed));write('closed-engine.json',dict(identity=identity,normal=True,engine_destroy_returned_zero=True,observed_all8_threads_exited=True));rows=[]
 for i,arm in enumerate(card['plan']):
  tokens=arm['prompt_tokens'];pf=(tokens+1023)//1024;f=pf+2;one=int(tokens%1024==1);t1=86*(2+one)
  h=dict(qualified=True,prequeue_bound=True,same_prepared_encoder_reproduces_prompt_and_IDs=True,markers_sent_to_Native=False,content_one_contribution_each_and_ordered=True,all_content_contribution_slices_equal=True,output_budget=4096,prompt_tokens=tokens,Native_prompt_token_ids_SHA256=arm['prompt_IDs_SHA256'],pending_thinking_prefix_open=False)
  rows.append(dict(request_id='request-'+str(i),admission=dict(history_render_audit=h,text_delivery='P1'),prompt_tokens=tokens,max_new_tokens=4096,native_finish_reason='eos_token',normal_reset_before=True,normal_reset_after=True,generated_token_ids=[10,11,1],actual_generated_tokens=3,actual_forward_attempts=f,actual_forward_completed=f,Native_continuation=[dict(kind='prefill',past=p,tokens=min(1024,tokens-p),total=min(p+1024,tokens))for p in range(0,tokens,1024)]+[dict(kind='decode_summary',calls=2,first_past=tokens,last_total=tokens+2)],r454_index=i,r454_mode=1,r454_profile='deep-hc0-fullpre-c1',r454_quiescent=True,r454_poisoned=False,r454_r366_fixed=0,r454_actual_T1_coverage=dict(passed=True,HC_groups=86,completed_forwards=f,one_token_prefill_forwards=one,decode_forwards=2,T1_HC_per_owner=t1,mode=1,r366_fixed=0),r454_actual_owner_counts=[dict(owner=o,created=86,ready=86,legacy=dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*f-t1,f32_t1=0),fullpre=dict(observed=86*f,eligible=t1,selected=t1,completed=t1,prefill=86*f-t1,f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0))for o in range(8)],backend_token_monotonic=[2.,3.,4.],started_monotonic=1.,Native_forward_host_summary=dict(prefill_calls=pf,prefill_s=.1,decode_calls=2,decode_s=.2),final_metadata=dict(recording=True,buffered_P0_text=False)))
 owners=[dict(device=i,changes=1,verified_mask=','.join(map(str,range(i*4,i*4+4))))for i in range(8)]
 total=sum(r['actual_forward_completed']for r in rows)
 write('owner.json',dict(identity=identity,completed=True,normal_Native_shutdown=True,engine_worker_shutdown_completed=True,observed_all8_threads_exited=True,loaded_key_count=67569,Native_SHA256='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1',HC_mode=0,fixed_profile=fixed,owner_affinity_after_weight_load=dict(actual_owner_report=dict(owners=owners)),owner_thread_observation=dict(status='OBSERVED',owners=[dict(TID=2000+i)for i in range(8)]),requests=rows,HC_dispatch_checked_forwards=total,layout_dispatch_checked_forwards=total))
class Response:
 def __init__(self,raw,stream):self.raw=raw;self.status=200;self.headers={'Content-Type':'text/event-stream'if stream else'application/json'};self.position=0
 def __enter__(self):return self
 def __exit__(self,*a):return False
 def read(self,n=-1):
  value=self.raw[self.position:]if n<0 else self.raw[self.position:self.position+n];self.position+=len(value);return value
 def __iter__(self):return iter(self.raw.splitlines(keepends=True))

def run_client(root,card,ff,private,identity,*,bad=False,bad_hits=False,args=None):
 folder=root/'results/Deep-fixed-profile-public-r457'/controller.RUN
 if not folder.exists():reports(folder,identity,card)
 if bad_hits:
  path=folder/'owner.json';q=json.loads(path.read_bytes());q['requests'][1]['r454_actual_owner_counts'][7]['fullpre']['completed']=0;path.write_text(json.dumps(q))
 inputs=fixture.assets(card,private);sent=[]
 def open_http(request,timeout):
  i=len(sent);sent.append(request);arm,raw,body,gold=inputs[i];assert request.data==raw and json.loads(request.data)==body
  content=('#### '+gold['gold_numeric'])if gold['suite']=='gsm8k'else('Answer: '+gold['gold_choice'])
  if bad:content='No valid answer in this recording wire'
  message=dict(role='assistant',content=content);usage=dict(prompt_tokens=arm['prompt_tokens'],completion_tokens=3,total_tokens=arm['prompt_tokens']+3)
  wire=('data: '+json.dumps(dict(id='request-'+str(i),choices=[dict(index=0,delta=message,finish_reason='stop')],usage=usage))+'\n\ndata: [DONE]\n\n').encode();return Response(wire,True)
 output=root/'client-output'
 if args is None:args=['--base-url','http://127.0.0.1:43882/v1','--runtime-dir',str(root/'results/Deep-fixed-profile-public-r457'),'--run',controller.RUN,'--expected-step',identity['actual_step'],'--source-sha',identity['source_manifest_SHA256'],'--config-sha',identity['provider_config_SHA256'],'--fixtures',str(ff),'--output-dir',str(output),'--deadline-unix',str(time.time()+500)]
 else:output=Path(args[args.index('--output-dir')+1])
 with patch.object(client,'assets',lambda card:fixture.assets(card,private)),patch.object(client.urllib.request,'urlopen',open_http),patch.object(client,'wait_listener',lambda *args:None):rc=client.main(args)
 q=json.loads((output/'result.json').read_bytes());assert q['completed']and q['passed']==(not(bad or bad_hits))and rc==(1 if bad or bad_hits else 0),q
 assert q['planned_HTTP']==2 and q['attempted_HTTP']==(1 if bad else 2)and q['completed_HTTP']==(1 if bad or bad_hits else 2)
 if bad:assert q['rows'][0]['score']['parse_failure']and not q['rows'][0]['score']['correct']
 return output/'result.json'
def main():
 cases=[];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());identity=dict(run=controller.RUN,actual_step='910374.99999',source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=time.time(),Native_seconds=5280)
 with tempfile.TemporaryDirectory()as tmp:
  private=copy_private(Path(tmp));card=json.loads((D/'FIXTURE_CARD.json').read_bytes());inputs=fixture.assets(card,private);assert[len(v[2]['messages'])for v in inputs]and[entry[0]['selected_ordinal']for entry in inputs]==[0,128]
  proof=preflight.check(D,Path(os.environ['DEEP457_PROVIDER']),m,local_files=private,cli_leaf=lambda d,m:dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False,experimental_profile='deep-hc0-fullpre-c1'))
  cases.append(dict(name='first_original_GSM_MMLU_fullbody_ids_gold_945_and962_pinned',passed=True,proof=proof,encoding_repeated=False))
  for scenario in['normal','bad-gold','bad-hit']:
   root=private/scenario;root.mkdir();ff=root/'fixture.json';ff.write_text(json.dumps(card));output=run_client(root,card,ff,private,identity,bad=scenario=='bad-gold',bad_hits=scenario=='bad-hit')
   if scenario=='normal':
    from public_deep_native.protocol import Run
    Run(dict(runtime_directory=str(root/'results/Deep-fixed-profile-public-r457'),**identity)).stop()
    summary=collect.collect(root,controller.RUN,identity['actual_step'],m['provider_source_SHA256'],output,time.time()+5);assert summary['external_requests']==2 and summary['Native_CAPI_calls']==7 and summary['normal8internalowner_destroy']
   cases.append(dict(name='actual_client_main_'+scenario+'_and_collector_if_normal',passed=True,HTTP_Native_reports_recording=True))
  from model_counts import validate
  row=json.loads((private/'normal/results/Deep-fixed-profile-public-r457'/controller.RUN/'owner.json').read_bytes())['requests'][1]
  for key,value in [('r454_mode',0),('actual_forward_completed',2)]:
   bad=copy.deepcopy(row);bad[key]=value
   try:validate(bad,1)
   except AssertionError:pass
   else:raise AssertionError('Wrong profile/forward accepted')
  command=controller.gpu_command();assert '--gres=dcu:8'in command and'01:30:00'in command and'-c32'in command and'--mem=128000M'in command and command[command.index('-w')+1]=='a02r4n04'
  child=controller.client_command('910374.99999',m,time.time());assert '01:25:00'in child and'--gres=none'in child and child[child.index('--fixtures')+1].endswith('FIXTURE_CARD.json')
  env=controller.gpu_environment(dict(DEEP_stale='x',SLURM_STEP_ID='old'),123.,m);assert env['DEEP_R457_MODEL_SOURCE_SHA']==m['provider_source_SHA256']and env['DEEP_R457_MODEL_GPUSTART']=='123.0'and'DEEP_stale'not in env and'SLURM_STEP_ID'not in env
  cases.append(dict(name='actual_controller_GPU_client_argv_namespace_and_fixed_contract',passed=True,actual_Popen_Slurm=False))
 with patch.object(controller,'verify',lambda digest:m),patch.object(preflight,'check',lambda *args:proof),contextlib.redirect_stdout(io.StringIO())as capture:
  assert controller.main(['--manifest-sha','recording','--publication-commit','0'*40,'--preflight-only'])==0
 assert json.loads(capture.getvalue())['proof']==proof
 import launch_once
 def fake_run(argv,**kwargs):
  assert argv==['python3','-B',str(D/'controller.py'),'--manifest-sha','recording','--publication-commit','0'*40,'--preflight-only']or argv in[['bash','-n',str(D/'inside.sh')],['bash','-n',str(D/'resource_wave_host_alias_outer.sh')]]
  return types.SimpleNamespace(returncode=0,stdout=json.dumps(dict(passed=True,source_only=True,GPU=0,Slurm=0,proof=proof)),stderr='')
 def metadata(argv,**kwargs):
  if argv[0]=='sacct':return ''
  if argv[:3]==['scontrol','show','job']:return 'JobState=RUNNING UserId=aclsylqidf(123) NodeList='+('a14r2n20'if argv[3]=='910407'else'a02r4n04')
  assert argv[:3]==['scontrol','show','hostnames'];return argv[3]
 with patch.object(launch_once,'verify',lambda digest:m),patch.object(launch_once.subprocess,'run',fake_run),patch.object(launch_once.subprocess,'check_output',metadata),patch.object(launch_once.subprocess,'Popen',side_effect=AssertionError('No real keeper or GPU may start in CPU consumer')),patch.object(sys,'argv',['launch_once.py','--manifest-sha','recording','--publication-commit','0'*40,'--preflight-only']),contextlib.redirect_stdout(io.StringIO())as capture:launch_once.main()
 result=json.loads(capture.getvalue());assert result['Slurm_submitted']==0 and result['HTTP']==2 and result['warms']==0 and result['keeper_argv'][result['keeper_argv'].index('-t')+1]=='01:35:00'
 cases.append(dict(name='actual_controller_and_launch_main_preflight_only_filehandoff',passed=True,Slurm_and_child_CLI_recording=True,Popen_forbidden=True))
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 print(json.dumps(dict(passed=True,cases=cases,actual_client_and_collect=True,actual_controller_command_environment=True,recording_HTTP_and_Native_reports=True,actual_Native_GPU_Model_or_HTTP=False,old962_or945_or938_replayed=False)))
if __name__=='__main__':main()
