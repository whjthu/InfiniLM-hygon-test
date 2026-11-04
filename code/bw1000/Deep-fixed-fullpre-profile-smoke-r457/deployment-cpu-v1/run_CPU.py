"""One necessary new smoke deployment CPU closure; no encoding/Native/weights."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Deep457-smoke-glue-CPU-v1');B.mkdir(exist_ok=False);started=time.monotonic()
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,weights=False,encoding_repeated=False,new_compiles=0,commands=[])
try:
 q['actual_CPU']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert all(os.environ.get(k)=='-1'for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mask=sorted(os.sched_getaffinity(0));assert len(mask)>=4;os.sched_setaffinity(0,set(mask[:4]))
 mf=H(D/'SOURCE_MANIFEST.json');assert mf==os.environ['DEEP457_CPU_SOURCE_SHA256'];q['source_manifest_SHA256']=mf;m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
 package=Path('/refactor/incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1');env=dict(os.environ,PYTHONPATH=':'.join(map(str,[D,package,Path('/refactor/incoming/public-Deep-native-provider-r427/source-v3'),Path('/old/offline-deps')])),DEEP457_PROVIDER=str(package))
 def run(argv,name):
  left=250-(time.monotonic()-started);assert left>0;p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=min(110,left));(B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr);q['commands'].append(dict(argv=argv,name=name,rc=p.returncode,stdout_SHA256=H(B/(name+'.stdout')),stderr_SHA256=H(B/(name+'.stderr'))));assert p.returncode==0,(name,p.stdout[-6000:],p.stderr[-6000:]);return json.loads(p.stdout)
 for shell in['inside.sh','resource_wave_host_alias_outer.sh']:
  p=subprocess.run(['bash','-n',str(D/shell)],capture_output=True,text=True,timeout=10);assert p.returncode==0,p.stderr
 q['new_glue']=run([sys.executable,'-B',str(D/'check_CPU.py')],'new-two-case-deployment-consumers')
 argv=[sys.executable,'-B','-m','public_deep_model','--config',str(D/'config.json'),'--config-sha',H(D/'config.json'),'--source-sha',m['provider_source_SHA256'],'--run','CPU.source-only','--actual-step',q['actual_CPU'],'--gpu-start-epoch','0']
 q['whole_CLI']=run(argv,'actual-public-CLI-n20-not-serving');assert q['whole_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False,experimental_profile='deep-hc0-fullpre-c1')
 assert q['new_glue']['passed']and not q['new_glue']['actual_Native_GPU_Model_or_HTTP']
 maps=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(s in line for s in['libinfinilm','libinfiniops','libinfiniccl','libinfinirt','_infinilm.cpython','libflash_attn'])];assert not maps;q['project_maps']=maps
 q.update(passed=True,old962_945_938_not_replayed=True,HTTP_planned=2,warmups=0,profile='deep-hc0-fullpre-c1',config_SHA256=H(D/'config.json'),provider_source_SHA256=m['provider_source_SHA256'])
except BaseException as error:q.update(passed=False,error=repr(error),traceback=traceback.format_exc())
finally:
 q.update(completed=True,wall_s=time.monotonic()-started);(B/'result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in['actual_CPU','completed','passed','error','wall_s']}))
if not q['passed']:raise SystemExit(1)
