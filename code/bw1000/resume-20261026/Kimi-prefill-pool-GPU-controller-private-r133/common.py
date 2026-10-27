import os,pathlib,json,datetime,subprocess,time,socket
ROOT=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');SRC=ROOT/'incoming/Kimi-prefill-pool-controller-r133';RES=json.loads((SRC/'resources.json').read_text());ACTIVE={'RUNNING','PENDING','COMPLETING','CONFIGURING'}
def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_bytes(json.dumps(d,indent=2).encode());tmp.replace(p)
def clean_env():
 env=dict(os.environ);removed=[]
 for k in list(env):
  if(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:removed.append(k);env.pop(k)
 return env,removed
def account(parent,name,env):
 lines=subprocess.check_output(['sacct','-S','2026-10-25','-j',parent,'-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=25).splitlines();rows=[s.split('|')for s in lines if len(s.split('|'))>4 and s.split('|')[1]==name];assert len(rows)<=1,(name,rows);return rows[0]if rows else None
def terminal(parent,name,expected,env,seconds=100):
 deadline=time.monotonic()+seconds
 while True:
  row=account(parent,name,env)
  if row:
   assert row[0]==expected,(row,expected)
   if row[2].split()[0].rstrip('+')not in ACTIVE:return row
  assert time.monotonic()<deadline,('terminal timeout',expected);time.sleep(2)
def verify_parent(g,env):
 raw=subprocess.check_output(['scontrol','show','job',g['parent'],'-o'],env=env,text=True,timeout=25);assert'UserId=aclsylqidf('in raw and'JobState=RUNNING'in raw;spec=next(v.split('=',1)[1]for v in raw.split()if v.startswith('NodeList='));nodes=subprocess.check_output(['scontrol','show','hostnames',spec],env=env,text=True,timeout=25).splitlines();assert set(g['nodes'])<=set(nodes);return raw
