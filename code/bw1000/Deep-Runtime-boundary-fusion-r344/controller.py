"""Exactly one two-node component batch. Published-source commit passed explicitly."""
import pathlib,subprocess,os,time,json,hashlib,sys,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Deep-Runtime-boundary-fusion-r344'
assert len(sys.argv)==2 and len(sys.argv[1])==40
H=R/'results/Deep344-control.json';assert not H.exists()
q=dict(sourceclaimcommit=sys.argv[1],completed=False,GPU_submitted=False,pre={},post={},steps={})
env=dict(os.environ)
for k in list(env):
 if (k.startswith(('SLURM_','PMI_','PMIX_','OMPI_')) and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:env.pop(k)
def save():H.write_text(json.dumps(q,indent=2)+'\n')
def idle(node,phase):
 a=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=Deep344-'+phase+'-'+node,'python3',str(R/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=150);v=dict(returncode=z.returncode,stdout=z.stdout,stderr=z.stderr);q[phase][node]=v;save();assert z.returncode==0,v
def account():
 return subprocess.check_output(['sacct','-j','910374','-S','2026-11-01','-n','-P','-o','JobIDRaw,JobName%70,State%30,ExitCode,NodeList%50'],text=True,env=env,timeout=25)
children=[];start=time.monotonic()
try:
 m=json.loads((S/'manifest.json').read_text())
 for n,h in m['files'].items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==h,n
 assert hashlib.sha256((R/'candidates/Deep-Runtime-boundary-fusion-r344/libDeep344.so').read_bytes()).hexdigest()=='296e95e83789625757a0ab744b5f2a0639699f2e144fb7f13ba7e0e1891952ab'
 raw=account();q['dedup_accounting']=raw;assert not any('|Deep344-GPU-'in line for line in raw.splitlines()),'existing GPU batch, never retry'
 for node in ['a02r4n04']:idle(node,'pre')
 for node in ['a02r4n04']:
  a=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=0','-N1','-n4','--ntasks-per-node=4','-c4','-w',node,'--gres=dcu:4','--mem=32000M','-t','00:30:00','--job-name=Deep344-GPU-'+node,'--export=ALL,WAVE_INSIDE_R2323=Deep-Runtime-boundary-fusion-r344/GPU2_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  f=(R/f'logs/Deep344-GPU-{node}-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append((node,p,f));q['steps'][node]=dict(client_PID=p.pid,argv=a);q['GPU_submitted']=True;save()
 while any(p.poll()is None for _,p,_ in children):
  assert time.monotonic()-start<1800,'batch timeout; Slurm step own hard limit remains'
  q['last_accounting']=account();q['elapsed_s']=time.monotonic()-start;save();time.sleep(10)
 for node,p,f in children:f.close();q['steps'][node]['returncode']=p.returncode
 q['last_accounting']=account();save()
 for node in ['a02r4n04']:idle(node,'post')
 q['results']=[str(p)for p in sorted((R/'results').glob('Deep344-*-rank*.json'))];q['completed']=all(p.returncode==0 for _,p,_ in children)and len(q['results'])==4
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
finally:q['elapsed_s']=time.monotonic()-start;save()
