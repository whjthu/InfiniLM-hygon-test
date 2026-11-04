"""Actual eight metadata/CLI consumers; explicit Slurm/Native report leaves only."""
from pathlib import Path
import contextlib,copy,hashlib,importlib.util,io,json,os,sys,tempfile,time
from unittest.mock import patch
D=Path(__file__).resolve().parent;PACKAGE=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def row_for(entry,*,cap=False):
 p=entry['prompt_tokens'];decode=4095 if cap else 1;one=int(p%1024==1);f=(p+1023)//1024+decode;t1=86*(one+decode);mode=entry['mode']
 return dict(request_id='recording-'+str(entry['logical_index']),r450_logical_index=entry['logical_index'],r450_case_id=entry['case_id'],r450_mode=mode,r450_r366_fixed=0,normal_reset_before=True,normal_reset_after=True,r450_quiescent=True,r450_poisoned=False,actual_forward_attempts=f,actual_forward_completed=f,generated_token_ids=[9]*4096 if cap else[9,1],max_new_tokens=4096,prompt_tokens=p,Native_continuation=[dict(kind='prefill',past=s,tokens=min(1024,p-s),total=min(s+1024,p))for s in range(0,p,1024)]+[dict(kind='decode_summary',calls=decode,first_past=p,last_total=p+decode)],r450_actual_owner_counts=[dict(owner=d,created=86,ready=86,legacy=dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-one-decode),f32_t1=0),fullpre=dict(observed=mode*86*f,eligible=mode*t1,selected=mode*t1,completed=mode*t1,prefill=mode*86*(f-one-decode),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0))for d in range(8)],r450_binding_proof=dict(actual_HC_groups_per_owner=86,HC0_fixed=True,providers={n:'/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'for n in ['deep_hc_fullpre_enable_r377']}),final_metadata=dict(buffered_P0_text=False))

def run():
 sys.path[:0]=[str(D),str(PACKAGE),'/refactor/incoming/public-Deep-native-provider-r427/source-v3']
 import preflight
 from public_deep_model.cli import main as actual_cli
 from coverage import validate
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());mf=sha(D/'SOURCE_MANIFEST.json');shards=json.loads((D/'SHARDS.json').read_bytes());allrows=[];values=[]
 def cli_leaf(deploy,manifest,shard):
  output=io.StringIO();config=Path(deploy)/shard['config']
  with contextlib.redirect_stdout(output):actual_cli(['--config',str(config),'--config-sha',sha(config),'--source-sha',manifest['provider_source_SHA256'],'--run','CPU.metadata-only','--actual-step',os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'--gpu-start-epoch','0'])
  q=json.loads(output.getvalue());assert q==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
  return dict(actual_public_CLI=q,actual_Python_module_function=True,container_transport_recorded_only=True)
 for j,shard in enumerate(shards):
  c=module('r450_metadata_controller_'+str(j),D/'controller.py');c.configure(j);table=json.loads((D/shard['plan']).read_bytes());allrows.extend(table['assigned'])
  with patch.object(preflight,'whole_cli',cli_leaf),contextlib.redirect_stdout(io.StringIO()):rc=c.main(['--shard',str(j),'--manifest-sha',mf,'--publication-commit','0'*40,'--preflight-only'])
  assert rc==0;current=c.verify(mf);gpu=c.gpu_command();client=c.client_command(str(shard['parent'])+'.12345',current,1000.)
  assert '--jobid='+str(shard['parent'])in gpu and'-c32'in gpu and'--gres=dcu:8'in gpu and'08:10:00'in gpu and gpu[gpu.index('-w')+1]==shard['node']
  assert '--jobid='+str(shard['parent'])in client and'--gres=none'in client and'-c2'in client and'07:50:00'in client and client[client.index('-w')+1]==shard['node']
  assert client[client.index('--deadline-unix')+1]=='29200.0' # original epoch1000 +service28200
  env=c.gpu_environment({},1000.,current);assert env['DEEP450_SHARD_ID']==str(j)and env['DEEP_R450_MODEL_CONFIG_SHA']==current['configs_SHA256'][str(j)]and env['WAVE_INSIDE_R2323'].endswith('/deployment-v1/inside.sh')and env['DEEP_R450_MODEL_DEPLOY_DIR'].endswith('/deployment-v1')
  for entry in table['assigned'][:2]+table['assigned'][-2:]:assert validate(row_for(entry),entry)['passed']
  # Actual launcher argparse/filehandoff; only child process creation is a CPU
  # transport leaf executing the corresponding actual controller entry here.
  launch=module('r450_metadata_launch_'+str(j),D/'launch_once.py');launch.c=c;real_run=launch.subprocess.run
  def child(command,**kw):
   if command[0]=='bash':return real_run(command,**kw)
   assert command[:2]==['python3','-B']and command[2]==str(D/'controller.py')and'--preflight-only'in command
   output=io.StringIO()
   with patch.object(preflight,'whole_cli',cli_leaf),contextlib.redirect_stdout(output):code=c.main(command[3:])
   return type('RecordedChild',(),dict(returncode=code,stdout=output.getvalue(),stderr=''))()
  with patch.object(launch.subprocess,'run',child),patch.object(launch.subprocess,'Popen',side_effect=AssertionError('CPU must not Popen keeper/GPU')),contextlib.redirect_stdout(io.StringIO()):code=launch.main(['--shard',str(j),'--manifest-sha',mf,'--publication-commit','0'*40,'--preflight-only'])
  assert code==0;values.append(dict(shard=j,parent=shard['parent'],node=shard['node'],actual_CLI=True,actual_controller_and_launch_main=True,first_pair=[r['logical_index']for r in table['assigned'][:2]],last_pair=[r['logical_index']for r in table['assigned'][-2:]],request_count=64,Slurm_or_Native=False))
 assert len(allrows)==512 and len({r['logical_index']for r in allrows})==512 and all(r==json.loads((D/'plans/shard0.json').read_bytes())['frozen512'][r['logical_index']]for r in allrows)
 from first_pair_gate import check
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);j=shards[0];folder=root/'results/Deep-broad-fewshot-fullpre-quality-r450'/j['run'];folder.mkdir(parents=True)
  table=json.loads((D/j['plan']).read_bytes());identity=dict(actual_step='910374.12345',source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['configs_SHA256']['0'])
  rows=[row_for(e,cap=i==0)for i,e in enumerate(table['assigned'][:2])]
  client=dict(completed_HTTP=2,rows=[dict(logical_index=e['logical_index'],actual_owner_coverage=validate(r,e),request_body_SHA256=e['request_body_SHA256'],correct=False,cap_hit=i==0,parse_failure=i==1)for i,(e,r)in enumerate(zip(table['assigned'][:2],rows))])
  watch=dict(shard=0,provider_source_SHA256=m['provider_source_SHA256'],deployment_source_SHA256=mf,actual_GPU='910374.12345',forced_cleanup=False,original_GPU_start_unix=1000.)
  ready=dict(identity=identity,Model_READY=True,Native_internal_owner_count=8,HC_mode=0);owner=dict(identity=identity,loaded_key_count=67569,Native_SHA256='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1',requests=rows)
  label=j['label']+'-'+j['run'];items=[(root/'results'/(label+'-watch.json'),watch),(root/'results'/(label+'-client/result.json'),client),(folder/'MODEL_READY.json',ready),(folder/'owner.json',owner)]
  for p,q in items:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(q))
  current=dict(m,source_manifest_SHA256=mf);gate=check(root,D,current);assert gate['passed']and not gate['quality_used_as_gate']and all(not r['observed_correct']for r in gate['first_pair_execution'])
  bad=copy.deepcopy(owner);bad['requests'][1]['r450_actual_owner_counts'][7]['fullpre']['completed']-=1;(folder/'owner.json').write_text(json.dumps(bad))
  try:check(root,D,current)
  except AssertionError:pass
  else:raise AssertionError('Missing true owner coverage rejection')
  (folder/'owner.json').write_text(json.dumps(owner));client['completed_HTTP']=1;(root/'results'/(label+'-client/result.json')).write_text(json.dumps(client));assert check(root,D,current)['pending']
 return dict(passed=True,actual_eight_CLI_and_metadata=values,union_global512_original_rows=True,first_and_last_pair_coverage_each_shard=True,firstpair_gate_ignores_gold_cap_parse=True,firstpair_missing_actual_owner_rejected=True,firstpair_pending_no_submission=True,recording_leaves=['Slurm child creation','Native owner/READY reports for gate'],old945_encoding_or_twoHTTP_consumer_replayed=False,GPU=0,Native_dlopen=False,Model_READY=False)
