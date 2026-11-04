"""Actual whole source CLI in the existing CPU-only container, before any GPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
def whole_cli(deploy,m):
 # The immutable production CLI verifies /refactor parent manifests. Use the
 # same existing namespace instead of faking those paths on the login host.
 old='/public/home/aclsylqidf/hygon-inference-20261013'
 inside='/refactor/incoming/Deep-public-prefill-chunk-2048-followthrough-r451/deployment-v1'
 argv=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--cleanenv','--home',old+'/build-deps/root:/root','--bind',old+':/old:ro','--bind',str(R)+':/refactor:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro',
  '--env','PYTHONPATH=/refactor/incoming/Deep-public-prefill-chunk-2048-followthrough-r451/source-v1:/refactor/incoming/public-Deep-native-provider-r427/source-v3:/old/offline-deps',
  '--env','CUDA_VISIBLE_DEVICES=-1,HIP_VISIBLE_DEVICES=-1,ROCR_VISIBLE_DEVICES=-1,USE_TORCH=0,USE_TF=0,LD_PRELOAD=',str(R/'containers/dtk26-sandbox-r1'),
  'python3','-B','-m','public_deep_model','--config',inside+'/config.json','--config-sha',m['config_SHA256'],'--source-sha',m['provider_source_SHA256'],'--run','source-only-validation','--actual-step','CPU.source-only','--gpu-start-epoch','0']
 env={k:v for k,v in os.environ.items()if k not in ['BASH_ENV','ENV','LD_PRELOAD','PYTHONSTARTUP','PYTHONINSPECT']and not k.startswith(('APPTAINER','SINGULARITY'))}
 result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=90);assert result.returncode==0,(result.stdout,result.stderr)
 q=json.loads(result.stdout);assert q==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 return dict(actual_whole_CLI=q,container_rocm=False,serving=False,Native_imported=False,argv=argv)
def check(deploy,provider,m,*,fixture_root=None,evidence_root=None,cli_leaf=None):
 deploy,provider=Path(deploy),Path(provider)
 from evidence import check as verify_cpu
 cpu=verify_cpu(deploy,provider,local_files=evidence_root)
 config=json.loads((deploy/'config.json').read_bytes());assert config['execute_authorized']and config['node']=='a02r4n12'and(config['Native_seconds'],config['service_seconds'])==(2580,2400)
 assert config['default_text_delivery']=='P1_when_stream_true_else_P0'and config['text_delivery_experiment']is True
 card=json.loads((deploy/'FIXTURE_CARD.json').read_bytes());assert [v['delivery']for v in card['plan']]==['P1']*6 and [v['mode']for v in card['plan']]==[0]*6 and [v['chunk']for v in card['plan']]==config['r451_chunks']==[1024,2048,1024,2048,2048,1024] and config['r451_fullpre_fixed']==0 and [v['warmup']for v in card['plan']]==[True,True,False,False,False,False]
 if fixture_root is not None:card['fixture_path']=str(Path(fixture_root)/card['local_untracked_fixture'])
 from fixture import body_and_gold
 inputs=[]
 for arm in card['plan']:
  raw,body,gold,h=body_and_gold(card,arm);assert body['max_tokens']==128 and body['stream']and body['chat_template_kwargs']=={'enable_thinking':False};inputs.append(h)
 assert len(set(inputs))==1 and card['original_prompt_tokens']==8489 and config['r451_output_budget']==128
 expected=json.loads((provider/'config.template.json').read_bytes());expected['execute_authorized']=True;assert config==expected
 inside=(deploy/'inside.sh').read_bytes();outer=(deploy/'resource_wave_host_alias_outer.sh').read_bytes();assert b'\r'not in inside+outer and b'\0'not in inside+outer
 assert b'python3 -B -m public_deep_model "${ARGS[@]}"'in inside and b'--serve'in inside and b'+2580-time.time()'in inside and b'--rocm'in outer
 actual=whole_cli(deploy,m)if cli_leaf is None else cli_leaf(deploy,m)
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 return dict(actual_CLI=actual,CPU_scopes=cpu,requests=6,warmups=2,formal=4,Native_imported=False,Slurm=0,Model_READY=False)
