"""Exact eight-shard metadata plus original actual public CLI; no serving."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()

def hostpath(value):
 p=Path(value);return R/str(p.relative_to('/refactor'))if str(p).startswith('/refactor/')else p

def assets(deploy):
 d=Path(deploy);pins=json.loads((d/'INPUT_PINS.json').read_bytes());out={}
 for name in ['CPU945','requests','cases']:
  p=hostpath(pins[name]['path']);assert sha(p)==pins[name]['SHA256'];out[name]=json.loads(p.read_bytes())
 e=out['CPU945'];assert e['completed']and e['passed']and e['actual_CPU']=='910407.945'and e['actual_encodings']==256 and e['actual_Deep_codec']and e['actual_public_protocol']and e['actual_prepared_FIFO']
 assert not any(e[k]for k in ['GPU','Native_dlopen','weights_loaded','Model_READY','HTTP'])and not e['final_modules']['project_or_SDK_maps']and not e['final_modules']['Torch_CUDA_initialized']
 assert hashlib.sha256(compact(out['requests']['logical_plan'])).hexdigest()==pins['frozen512_compact_SHA256']
 return pins,out

def whole_cli(deploy,m,shard):
 d=Path(deploy);old='/public/home/aclsylqidf/hygon-inference-20261013';source='/refactor/incoming/Deep-broad-fewshot-fullpre-quality-r450'
 from controller import incoming_relative
 config='/refactor/incoming/'+incoming_relative(d)+'/'+shard['config']
 argv=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--cleanenv','--home',old+'/build-deps/root:/root','--bind',old+':/old:ro','--bind',str(R)+':/refactor:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro','--env','PYTHONPATH='+source+':/refactor/incoming/public-Deep-native-provider-r427/source-v3:/old/offline-deps','--env','CUDA_VISIBLE_DEVICES=-1,HIP_VISIBLE_DEVICES=-1,ROCR_VISIBLE_DEVICES=-1,USE_TORCH=0,USE_TF=0,LD_PRELOAD=',str(R/'containers/dtk26-sandbox-r1'),'python3','-B','-m','public_deep_model','--config',config,'--config-sha',m['configs_SHA256'][str(shard['index'])],'--source-sha',m['provider_source_SHA256'],'--run','source-only-validation','--actual-step','CPU.source-only','--gpu-start-epoch','0']
 env={k:v for k,v in os.environ.items()if k not in ['BASH_ENV','ENV','LD_PRELOAD','PYTHONSTARTUP','PYTHONINSPECT']and not k.startswith(('APPTAINER','SINGULARITY'))}
 p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=90);assert p.returncode==0,(p.stdout,p.stderr);q=json.loads(p.stdout)
 assert q==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
 return dict(actual_public_CLI=q,argv=argv,container_rocm=False,serving=False)

def check(deploy,provider,m,shard,*,cli_leaf=None):
 d=Path(deploy);provider=Path(provider);pins,data=assets(d);index=shard['index']
 assert m['provider_source_SHA256']==pins['provider_source_SHA256']==sha(provider/'SOURCE_MANIFEST.json')and pins['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
 config=json.loads((d/shard['config']).read_bytes());plan=json.loads((d/shard['plan']).read_bytes());assert sha(d/shard['config'])==m['configs_SHA256'][str(index)]and sha(d/shard['plan'])==m['plans_SHA256'][str(index)]
 assert config['execute_authorized']==m['GPU_GO']and config['node']==shard['node']and config['Native_seconds']==28800 and config['service_seconds']==28200 and config['HC_mode']==config['r450_r366_fixed']==0
 assert config['r450_plan']==plan['assigned']and config['r450_frozen512']==plan['frozen512']==data['requests']['logical_plan']and len(plan['assigned'])==64 and plan['warmups']==0
 assert config['rows']==config['PP']==1 and config['TP']==8 and config['context']==32768 and config['Native_SHA256']==pins['Native_SHA256']
 from_root=data['CPU945']['cases'];ids=[e for suite in ['gsm8k','mmlu']for e in[e for e in from_root if e['suite']==suite][index::8]];lookup={(r['case_id'],r['mode']):r for r in plan['frozen512']}
 expected=[lookup[(e['case_id'],mode)]for j,e in enumerate(ids)for mode in([0,1]if j%2==0 else[1,0])];assert plan['assigned']==expected and len(ids)==32 and sum(e['suite']=='gsm8k'for e in ids)==16
 assert [r['mode']for r in expected[:2]]==[0,1]
 inside=(d/'inside.sh').read_bytes();outer=(d/'resource_wave_host_alias_outer.sh').read_bytes();assert b'\r'not in inside+outer and b'\0'not in inside+outer and b'+28800-time.time()'in inside and b'--rocm'in outer
 cli=whole_cli(d,m,shard)if cli_leaf is None else cli_leaf(d,m,shard)
 assert not any(n in sys.modules for n in ['_infinilm','flash_attn'])
 return dict(passed=True,shard=index,node=shard['node'],parent=shard['parent'],questions=32,HTTP=64,warmups=0,first_pair=[r['logical_index']for r in expected[:2]],last_pair=[r['logical_index']for r in expected[-2:]],actual_CLI=cli,CPU945_consumed=True,Native_dlopen=False,Model_READY=False)
