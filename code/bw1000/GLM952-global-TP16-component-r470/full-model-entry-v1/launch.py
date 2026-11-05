"""Non-executing HOST+samecontainer metadata/loader preflight; then once keeper."""
from pathlib import Path
import argparse,ast,json,os,re,socket,subprocess,time
from shared import D,R,OUT,LABEL,H,verify

def preflight(pin):
 m,cpu,inp,plan=verify(pin)
 for p in D.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 for p in D.glob('*.sh'):
  assert b'\r'not in p.read_bytes();subprocess.run(['bash','-n',str(p)],check=True,timeout=8)
 # No Runtime/Native/SDK function is called, no weights are opened. ldd child
 # initialization is UNKNOWN rather than a claim that no constructor ran.
 code="""import hashlib,json,sys
from pathlib import Path
from shared import verify,H,D
m,cpu,inp,plan=verify(sys.argv[1])
import torch,safetensors,tokenizers
assert not torch.cuda.is_initialized()
assert len(plan['requests'])==2
print(json.dumps(dict(passed=True,Native_calls=0,SDK_calls=0,weight_open=0,CUDA_initialized=torch.cuda.is_initialized(),private_plan_SHA256=inp['private_plan']['SHA256'])))
"""
 env_script='source '+str(D/'ENV.sh')+'\nunset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL\nexport CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""\ncd /refactor/incoming/GLM952-global-TP16-component-r470/full-model-entry-v1\n/usr/bin/python3 -B -c '+__import__('shlex').quote(code)+' '+pin+'\nldd -r '+__import__('shlex').quote(cpu['library'])+'\n'
 args=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--home','/public/home/aclsylqidf/hygon-inference-20261013/build-deps/root:/root','--bind','/public/home/aclsylqidf/hygon-inference-20261013:/old:ro','--bind',str(R)+':/refactor','--bind',str(R)+':'+str(R)+':ro','--bind','/opt/hyhal:/opt/hyhal:ro','--bind','/public/home/aclsylqidf/shared:/shared:ro',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-e','-c',env_script]
 z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90);assert z.returncode==0,(z.returncode,z.stderr[-500:]);assert 'not found'not in (z.stdout+z.stderr).lower()and 'undefined symbol'not in(z.stdout+z.stderr).lower();rows=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert len(rows)==1 and rows[0]['passed']and not rows[0]['CUDA_initialized'];return dict(passed=True,samecontainer=rows[0],ldd_combined_SHA256=__import__('hashlib').sha256((z.stdout+'\n'+z.stderr).encode()).hexdigest(),ldd_child_initialization='UNKNOWN',GPU=False,Slurm_submissions=0)

def main(publication,pin,only):
 assert re.fullmatch('[0-9a-f]{40}',publication);verify(pin);intent=R/'results'/(LABEL+'-intent.json');launch=R/'results'/(LABEL+'-launch.json');watch=R/'results'/(LABEL+'-watch.json');assert not any(p.exists()for p in[intent,launch,watch,OUT]);z=subprocess.run(['sacct','-j','910371,910407','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15);assert z.returncode==0 and not any(LABEL in line for line in z.stdout.splitlines())
 pre=R/'results'/(LABEL+'-HOST-preflight.json')
 if only:
  assert not pre.exists();q=preflight(pin);q.update(HOST=socket.gethostname(),publication=publication,GPU_manifest_SHA256=pin,epoch=time.time());pre.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q));return
 q=json.loads(pre.read_bytes());assert q['passed']and q['publication']==publication and q['GPU_manifest_SHA256']==pin and time.time()-q['epoch']<1800
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c2','--mem=8192M','--gres=none','-w','a14r2n20','-t','04:10:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'env','GLM470_GPU_MF='+pin,'bash',str(D/'keeper.sh')]
 with intent.open('x')as f:json.dump(dict(once=True,source_publication=publication,GPU_manifest_SHA256=pin,claim_SHA256=H(D/'GPU_RESOURCE_CLAIM.json'),HOST=socket.gethostname(),epoch=time.time(),Model_execution=True,Model_qualified=False,planned=2,HTTP=0,argv=args),f)
 with(R/'logs'/(LABEL+'-keeper-srun.log')).open('xb')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(HOST=socket.gethostname(),owned_keeper_PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],GPU=None,intent=str(intent));launch.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--publication',required=True);a.add_argument('--manifest-sha',required=True);a.add_argument('--preflight-only',action='store_true');v=a.parse_args();main(v.publication,v.manifest_sha,v.preflight_only)
