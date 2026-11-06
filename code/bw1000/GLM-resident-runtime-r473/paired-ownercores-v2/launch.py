"""Non-executing HOST+samecontainer metadata/loader preflight; then once keeper."""
from pathlib import Path
import argparse,ast,json,os,re,socket,subprocess,time
from shared import D,R,OUT,LABEL,H,verify

def preflight(pin):
 from check_deadlines_CPU import verify as verify_deadlines
 verify_deadlines(D)
 m,cpu,inp,plan=verify(pin)
 for p in D.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 for p in D.glob('*.sh'):
  assert b'\r'not in p.read_bytes();subprocess.run(['bash','-n',str(p)],check=True,timeout=8)
 # Reuse the actual successful sameENV/container metadata probe. The old v1
 # failure remains a failure; no repeated expensive Python imports are required.
 from shared import actual
 b=json.loads((D/'PREFLIGHT_BINDING.json').read_bytes());assert H(D/'ENV.sh')==b['ENV_SHA256']and cpu['library_SHA256']==b['Native_SHA256'];P=actual(b['diagnostic_source']);assert H(P/'SOURCE_MANIFEST.json')==b['diagnostic_manifest_SHA256'];dm=json.loads((P/'SOURCE_MANIFEST.json').read_bytes());assert all(H(P/n)==h for n,h in dm['files_SHA256'].items());result=json.loads(actual(b['diagnostic_result']).read_bytes());assert result['passed']and result['group_gone']and not result['terminal_UNKNOWN']and not result['timeout']and result['source_manifest_SHA256']==b['diagnostic_manifest_SHA256']
 streams={}
 for kind in ['stdout','stderr']:
  item=b['diagnostic_'+kind];p=actual(item['path']);assert p.stat().st_size==item['bytes']and H(p)==item['SHA256'];streams[kind]=p.read_text()
 newstreams={}
 for kind,item in b['new_Native_ldd_streams'].items():
  lp=actual(item['path']);assert H(lp)==item['SHA256']and lp.stat().st_size==item['bytes'];newstreams[kind]=lp.read_text()
 newcombined=newstreams['stdout']+'\n'+newstreams['stderr'];assert 'not found'not in newcombined and 'undefined symbol'not in newcombined
 combined=streams['stdout']+'\n'+streams['stderr'];assert 'not found'not in combined.lower()and 'undefined symbol'not in combined.lower();markers=[]
 for line in streams['stdout'].splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  if 'phase'in v:markers.append(v)
 phases=[v['phase']for v in markers];required=['container_bash_entry','source_ENV_before','source_ENV_after','Python_before','Python_entry','verify_before','verify_after_torch_before','torch_after_safetensors_before','safetensors_after_tokenizers_before','tokenizers_after','Python_after_ldd_before','ldd_after'];position=[phases.index(v)for v in required];assert position==sorted(position);torchrow=next(v for v in markers if v['phase']=='torch_after_safetensors_before');assert torchrow['CUDA_initialized']is False
 return dict(passed=True,qualified_environment_reused=True,original_diagnostic_request=b['original_request'],ENV_SHA256=b['ENV_SHA256'],Native_SHA256=b['Native_SHA256'],diagnostic_stdio_SHA256={k:b['diagnostic_'+k]['SHA256']for k in ['stdout','stderr']},GPU=False,Slurm_submissions=0,Native_calls=0,SDK_explicit_calls=0,weight_open=0,ldd_child_initialization='UNKNOWN',original_v1_preflight_passed=False)

def main(publication,pin,only):
 assert re.fullmatch('[0-9a-f]{40}',publication);verify(pin);intent=R/'results'/(LABEL+'-intent.json');launch=R/'results'/(LABEL+'-launch.json');watch=R/'results'/(LABEL+'-watch.json');assert not any(p.exists()for p in[intent,launch,watch,OUT]);z=subprocess.run(['sacct','-j','910371,910407','-S','2026-11-05','-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15);assert z.returncode==0 and not any(LABEL in line for line in z.stdout.splitlines())
 pre=R/'results'/(LABEL+'-HOST-preflight.json')
 if only:
  assert not pre.exists();q=preflight(pin);q.update(HOST=socket.gethostname(),publication=publication,GPU_manifest_SHA256=pin,epoch=time.time());pre.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q));return
 q=json.loads(pre.read_bytes());assert q['passed']and q['publication']==publication and q['GPU_manifest_SHA256']==pin and time.time()-q['epoch']<1800
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c2','--mem=8192M','--gres=none','-w','a14r2n20','-t','05:10:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'env','GLM470_GPU_MF='+pin,'bash',str(D/'keeper.sh')]
 with intent.open('x')as f:json.dump(dict(once=True,source_publication=publication,GPU_manifest_SHA256=pin,claim_SHA256=H(D/'GPU_RESOURCE_CLAIM.json'),HOST=socket.gethostname(),epoch=time.time(),Model_execution=True,Model_qualified=False,planned=10,HTTP=0,argv=args),f)
 with(R/'logs'/(LABEL+'-keeper-srun.log')).open('xb')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(HOST=socket.gethostname(),owned_keeper_PID=p.pid,PGID=p.pid,startticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],GPU=None,intent=str(intent));launch.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--publication',required=True);a.add_argument('--manifest-sha',required=True);a.add_argument('--preflight-only',action='store_true');v=a.parse_args();main(v.publication,v.manifest_sha,v.preflight_only)
