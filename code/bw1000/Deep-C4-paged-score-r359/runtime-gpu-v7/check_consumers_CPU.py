"""Exercise actual future result consumer/plan without Runtime or GPU libraries."""
from pathlib import Path
import importlib.util,json,hashlib,os,numpy as np,copy
S=Path(__file__).resolve().parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
driver=load('driver359_CPU',S/'GPU_driver.py')
P=Path('/refactor/incoming/Deep-C4-paged-score-r359-source-cpu-v2');oracle=load('oracle359_CPU',P/'reference.py')
plan=driver.plan();assert len(plan)==52
assert all([p['mode']for p in plan if p['past']==past and p['phase']=='measurement']==[0,1,1,0]*5 for past in [8496,8575])
def field(a):return dict(raw=a.view(np.uint8).reshape(-1).copy(),shape=a.shape,dtype=-1)
def fixtures(past):
 after=(past+1)//4;raw=np.full((128,8448),0xa5,np.uint8)
 for slot in range(after):
  page,i=divmod(slot,64);raw[page,i*128:(i+1)*128]=56;raw[page,8192+i*4:8196+i*4]=np.array([1],np.float32).view(np.uint8)
 f={0:field(np.full((1,8,128),56,np.uint8)),1:field(np.ones((1,8),np.float32)),2:field(np.full((1,128),56,np.uint8))if after>past//4 else None,3:field(np.ones((1,1),np.float32))if after>past//4 else None,4:field(np.full((1,after),8192,np.float32)),5:field(np.arange(512,dtype=np.int64).reshape(1,512)),6:field(raw),7:field(np.zeros(64,np.uint8))}
 return [dict(rank=i,past=past,fields=copy.deepcopy(f))for i in range(8)]
checks=[]
for past in [8496,8575]:
 proof=fixtures(past);r=driver.numeric_all(proof,oracle,None);assert len(r)==8 and all(v['CPU_FP8_score_NRMSE']==0 for v in r)
 assert all(v['TopK_overlap']==512 for v in driver.numeric_all(proof,oracle,copy.deepcopy(proof)))
 checks.append('actualconsumer8rank_F32score_cacheboundary_TopK_and_state_'+str(past))
for kind in ['nonfinite','duplicate_ids','bad_fp8','state_changed']:
 good=fixtures(8575);bad=copy.deepcopy(good)
 if kind=='nonfinite':bad[0]['fields'][4]['raw'].view(np.float32)[0]=float('nan')
 if kind=='duplicate_ids':bad[0]['fields'][5]['raw'].view(np.int64)[1]=0
 if kind=='bad_fp8':bad[0]['fields'][6]['raw'][0]=127
 if kind=='state_changed':bad[0]['fields'][7]['raw'][0]=1
 try:driver.numeric_all(bad,oracle,good)
 except (AssertionError,ValueError):checks.append('reject_'+kind)
 else:raise AssertionError('Consumer accepted '+kind)
R=Path('/refactor');receipt=R/'candidates/Deep-C4-paged-score-r359-runtime-link-v5/CPU-link.json';b=receipt.read_bytes();assert hashlib.sha256(b).hexdigest()=='4878209aa64f85b92a96bc8cb5cad606a1a814ab9502677d580e3953bee96f5d'
d=json.loads(b);assert d['completed']and hashlib.sha256(Path(d['library']).read_bytes()).hexdigest()==d['library_SHA256']=='a4e6208eef89bd63045c01bcab5ccc358e6a36cd6ea034aa286b2934ba82d91b'
q=dict(completed=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,Native_dlopen=False,checks=checks,driver_SHA256=hashlib.sha256((S/'GPU_driver.py').read_bytes()).hexdigest(),scope='Actual plan and result consumer exercised on known F32 page fixtures/negative mutations; no Runtime/CCL/loader/GPU executed',CPU723_SHA256=hashlib.sha256(b).hexdigest())
out=R/'candidates/Deep-C4-paged-score-r359-runtime-gpu-v7-CPU';out.mkdir(exist_ok=False);(out/'CPU-consumers.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
