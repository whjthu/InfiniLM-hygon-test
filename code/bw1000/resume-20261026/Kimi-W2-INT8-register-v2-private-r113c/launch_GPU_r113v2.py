"""Only unique launcher after root publishes exactsource and writes GO; never run by staging."""
import pathlib,json,hashlib,subprocess,os,re
D=pathlib.Path(__file__).resolve().parent;r=D.parent.parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=D/'GPU-manifest-source-r113v2b.json';go=json.loads((D/'GPU-GO-after-source-publication-r113v2b.json').read_bytes());assert go['authorized_by']=='parent-root'and go['manifest_SHA256']==sha(m)and re.fullmatch('[0-9a-f]{40}',go['Git_publication_SHA'])
record=r/'results/Kimi-W2-INT8-register-v2-r113c-keeper-launch.json';assert not record.exists();meta=json.loads(m.read_bytes())
for p,h in meta['source_SHA256'].items():assert sha(p.replace('/refactor',str(r),1))==h,p
prior=subprocess.check_output(['squeue','--steps','-j','910407','-h','-o','%i|%j|%N'],text=True,timeout=20).splitlines();numeric=[x for x in prior if x.split('|')[0].split('.')[-1].isdigit()];assert not numeric,numeric
raw=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert'UserId=aclsylqidf('in raw and'JobState=RUNNING'in raw and'a14r2n20'in raw
# Single c1/2GiB CPUkeeper, future16cards375/n08n09 only. Nootherparent touched.
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))}
a=['srun','--jobid=910407','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=2048M','-w','a14r2n20','-t','00:25:00','--job-name=Kimi-W2-INT8-register-v2-keeper-r113c','--output='+str(r/'logs/Kimi-W2-INT8-register-v2-keeper-r113c-%J.log'),'python3',str(D/'keeper_r113v2.py')]
f=(r/'logs/Kimi-W2-INT8-register-v2-keeper-r113c-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);q={'PID':p.pid,'args':a,'prior_numeric_CPU':numeric,'publication':go['Git_publication_SHA'],'manifest_SHA256':sha(m),'GPU_intent':'onefixed16card fullMoEcomponent historicalT8/T64+constructed896expertT64; noModel'};record.write_text(json.dumps(q));print(json.dumps(q))
