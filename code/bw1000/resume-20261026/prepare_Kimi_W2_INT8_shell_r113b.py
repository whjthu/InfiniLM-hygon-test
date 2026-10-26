import pathlib,json,hashlib,ast,base64
r=pathlib.Path(__file__).resolve().parent;d=r/'Kimi-W2-INT8-private-r113';old=r/'Kimi-W2-pair-private-r9280';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();new=[]
def put(n,s):
 p=d/n;assert not p.exists(),n;b=s if isinstance(s,bytes)else s.encode('utf8');b=b.replace(b'\r\n',b'\n')if n.endswith(('.py','.sh'))else b
 if n.endswith('.sh'):assert b'\r'not in b
 if n.endswith('.py'):ast.parse(b)
 p.write_bytes(b);new.append(n);return p
h=(old/'expected-headers-r9125.json').read_bytes();assert hashlib.sha256(h).hexdigest()=='40790f3859efe56852d95e7b872c666d95f009db410276f4ca018f3c360c9460';put('expected-headers-r9125.json',h)
put('resources-r113b.json',(old/'resources.json').read_bytes())
s=(old/'common.py').read_text().replace("SRC=ROOT/'incoming/Kimi-W2-pair-r9280'","SRC=ROOT/'incoming/Kimi-W2-INT8-r113'").replace("SRC/'resources.json'","SRC/'resources-r113b.json'");put('common_r113b.py',s)
s=(old/'keeper_r9280.py').read_text();s=s.replace('import common,','import common_r113b as common\nimport ');s=s.replace('Kimi-W2-pair-r9280','Kimi-W2-INT8-r113b').replace('GPU-manifest-source.json','GPU-manifest-source-r113b.json').replace('GPU-manifest.json','GPU-manifest-r113b.json').replace('GPU-GO-after-source-publication.json','GPU-GO-after-source-publication-r113b.json')
a=s.index(" w['CPUpreflight_reused']=");z=s.index(";w['fresh_before']=idle('before')",a)
s=s[:a]+" w['CPUpreflight_reused']={'actual_handle':'910375.741','scope':'existing identicalRuntime/loader importenvironment; no newCPU gate'};w['CPU_engineering_closure']={'actual_handle':'910375.751','CPU_only':True,'GPU':False,'receipt_SHA256':'5708702fc95b6e5a3d873c2dbc4a6af3b2f4b52509bcb00647a4d0e089383c6b','I8_ISA_CPU':'910375.750 sixW2+onewarmatom; noF64'}"+s[z:]
s=s.replace('WAVE_INSIDE_R2323=Kimi-W2-INT8-r113b/GPU_inside.sh','WAVE_INSIDE_R2323=Kimi-W2-INT8-r113/GPU_inside_r113b.sh')
s=s.replace("len(owner['arms'])==16 and len(owner['profile'])==0 and len(owner['current_branch'])==2","len(owner['arms'])==24 and len(owner['profile'])==0 and len(owner['current_branch'])==3 and owner['INT8_atom_CPU_integer_oracle_all256_pass']")
s=s.replace("branch['W2_original_OFF_calls']==34 and branch['W2_pair_ON_calls']==34","branch['W2_original_OFF_calls']==0 and branch['W2_pair_ON_calls']==34 and branch['I8_OFF_calls']==34 and branch['I8_ON_calls']==34")
s=s.replace('KIMI_W2_PAIR_R9280_NORMAL','KIMI_W2_INT8_R113_NORMAL')
# Audit scenario labels and original ascending16 reduction counters from all owners.
needle="  log=(r/'logs'"
s=s.replace(needle,"   assert [(b['T'],b['constructed_low_reuse']) for b in owner['current_branch']]==[(8,False),(64,False),(64,True)]\n"+needle)
put('keeper_r113b.py',s)
s=(old/'GPU_inside.sh').read_text();s=s.replace('bash -n /refactor/incoming/Kimi-W2-pair-r9280/GPU_inside.sh','bash -n /refactor/incoming/Kimi-W2-INT8-r113/GPU_inside_r113b.sh').replace('export INFINILM_KIMI_W2_PAIR_R9280=0','export INFINILM_KIMI_W2_PAIR_R9280=1 INFINILM_KIMI_W2_INT8_R113=0 INFINILM_KIMI_W2_CONSTRUCTED_R113=0')
a=s.index('timeout --kill-after=20s 1100 python');s=s[:a]+'''timeout --kill-after=20s 1100 python -B -u -c 'import os,sys;sys.path.insert(0,"/refactor/incoming/Kimi-W2-INT8-r113");import run_GPU_r113;run_GPU_r113.run("/refactor/incoming/Kimi-W2-INT8-r113/GPU-manifest-r113b.json","/refactor/results/Kimi-W2-INT8-r113b-"+os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"]+"-node"+os.environ["SLURM_PROCID"]+".json")'
''';put('GPU_inside_r113b.sh',s)
launcher='''"""Only unique launcher after root publishes exactsource and writes GO; never run by staging."""
import pathlib,json,hashlib,subprocess,os,re
D=pathlib.Path(__file__).resolve().parent;r=D.parent.parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=D/'GPU-manifest-source-r113b.json';go=json.loads((D/'GPU-GO-after-source-publication-r113b.json').read_bytes());assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==sha(m)and re.fullmatch('[0-9a-f]{40}',go['Git_publication_SHA'])
record=r/'results/Kimi-W2-INT8-r113b-keeper-launch.json';assert not record.exists();meta=json.loads(m.read_bytes())
for p,h in meta['source_SHA256'].items():assert sha(p.replace('/refactor',str(r),1))==h,p
prior=subprocess.check_output(['squeue','--steps','-j','910407','-h','-o','%i|%j|%N'],text=True,timeout=20).splitlines();numeric=[x for x in prior if x.split('|')[0].split('.')[-1].isdigit()];assert not numeric,numeric
raw=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert'UserId=aclsylqidf('in raw and'JobState=RUNNING'in raw and'a14r2n20'in raw
# Single c1/2GiB CPUkeeper, future16cards375/n08n09 only. Nootherparent touched.
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))}
a=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=2048M','-w','a14r2n20','-t','00:25:00','--job-name=Kimi-W2-INT8-keeper-r113b','--output='+str(r/'logs/Kimi-W2-INT8-keeper-r113b-%J.log'),'python3',str(D/'keeper_r113b.py')]
f=(r/'logs/Kimi-W2-INT8-keeper-r113b-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);q={'PID':p.pid,'args':a,'prior_numeric_CPU':numeric,'publication':go['Git_publication_SHA'],'manifest_SHA256':sha(m),'GPU_intent':'onefixed16card fullMoEcomponent historicalT8/T64+constructed896expertT64; noModel'};record.write_text(json.dumps(q));print(json.dumps(q))
''';put('launch_GPU_r113b.py',launcher)
m=json.loads((d/'GPU-manifest-source-r113.json').read_text());m.update(GPU_GO=False,GPU_started=False,root_ordinary_push_required_before_GPU=True,source_manifest_supersedes='GPU-manifest-source-r113.json metadata/dependencyclosure only, originalretained',original_manifest_SHA256=sha(d/'GPU-manifest-source-r113.json'),unique_launcher='/refactor/incoming/Kimi-W2-INT8-r113/launch_GPU_r113b.py',keeper='/refactor/incoming/Kimi-W2-INT8-r113/keeper_r113b.py',GPU_inside='/refactor/incoming/Kimi-W2-INT8-r113/GPU_inside_r113b.sh',runtime_manifest='/refactor/incoming/Kimi-W2-INT8-r113/GPU-manifest-r113b.json',GO_file='/refactor/incoming/Kimi-W2-INT8-r113/GPU-GO-after-source-publication-r113b.json')
m['Model_exact_flags']['INFINILM_KIMI_W2_PAIR_R9280']='1 fixed in botharms, paircounter34 OFFarmcalls';m['Model_exact_flags']['INFINILM_KIMI_W2_INT8_R113']='OFF/ON controlledbyactualrunner,34each';m['Model_exact_flags']['INFINILM_KIMI_W2_CONSTRUCTED_R113']='scenario0/1=0, scenario2=1 botharms';m['metadata_erratum']='OldsealedmanifestW2_PAIR default0/perarmOFFON text was inheritedr9280description; actualclosedwrapperalwaysPAIR1 andI8separateflag. No math recompile.'
m['source_SHA256'].update({'/refactor/incoming/Kimi-W2-INT8-r113/'+n:sha(d/n)for n in new});put('GPU-manifest-source-r113b.json',json.dumps(m,indent=2)+'\n')
proof={'status':'MECHANICAL_CONSUMER_DEPENDENCY_AND_UNIQUE_LAUNCHER_READY_GPU_NOT_AUTHORIZED','GPU':False,'new_CPU_job':False,'new_compile':False,'old_sealed_manifest_SHA256':sha(d/'GPU-manifest-source-r113.json'),'new_manifest_SHA256':sha(d/'GPU-manifest-source-r113b.json'),'expectedheaders_SHA256':sha(d/'expected-headers-r9125.json'),'consumer_unchanged_SHA256':sha(d/'run_GPU_r113.py'),'launcher_SHA256':sha(d/'launch_GPU_r113b.py'),'keeper_SHA256':sha(d/'keeper_r113b.py'),'GPU_inside_SHA256':sha(d/'GPU_inside_r113b.sh'),'resource':'GPU16card375n08n09/c32/128000MiB/1200hard1100inner; CPUkeeper407n20/c1/2048MiB','future_oracle':'insideoriginalcomponentwarm,noindependentGPUorCPUgate','root_GO_before_launch':True,'producer_SHA256':sha(__file__),'all_new_source_SHA256':{n:sha(d/n)for n in new},'reused_framework_SHA256':{n:sha(old/n)for n in ['common.py','keeper_r9280.py','GPU_inside.sh','resources.json','expected-headers-r9125.json']}}
put('source-producer-r113b.json',json.dumps(proof,indent=2)+'\n');index={'private':str(d),'newfiles_SHA256':{n:sha(d/n)for n in new},'originalsource_index_SHA256':sha(d/'source-producer-index-r113.json'),'producer_SHA256':sha(__file__)};put('source-producer-index-r113b.json',json.dumps(index,indent=2)+'\n')
# Stage only: no srun, no importtorch, no compile, no key/nonce access.
files={n:base64.b64encode((d/n).read_bytes()).decode()for n in new}
code="import pathlib,json,base64,hashlib\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014';D=r/'incoming/Kimi-W2-INT8-r113';assert D.is_dir()\n"
code+="out={}\nfor n,v in "+repr(files)+".items():\n p=D/n;b=base64.b64decode(v)\n if p.exists():assert p.read_bytes()==b,n\n else:p.write_bytes(b)\n out[n]=hashlib.sha256(p.read_bytes()).hexdigest()\n"
code+="print(json.dumps({'staged_SHA256':out,'GPU':False,'new_CPU_job':False,'compile':False,'GO_file_created':False}))\n";(r/'stage_Kimi_W2_INT8_shell_r113b.py').write_text(code)
print(json.dumps({k:proof[k]for k in ['new_manifest_SHA256','expectedheaders_SHA256','consumer_unchanged_SHA256','launcher_SHA256','keeper_SHA256','GPU_inside_SHA256','producer_SHA256']}))
