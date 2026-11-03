"""One <=300s actual two-fixture encode + unchanged Native graph + mixed entry CPU."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,traceback

def main():
 d=Path(__file__).resolve().parent;step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];root=Path('/refactor');out=root/'candidates'/('Deep-PD-performance-r445-CPU-'+step);out.mkdir(exist_ok=False)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
 q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,weights_opened=False,actual_CPU=step,phases={})
 try:
  mf=d/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP_PD_CPU_SOURCE_SHA'];source=json.loads(mf.read_bytes());assert not source['GPU_GO']and all(sha(d/n)==h for n,h in source['files_SHA256'].items());q['source_manifest_SHA256']=sha(mf)
  for name,args in [('static_binding',['CPU_static.py']),('actual_fixture',['CPU_fixture.py',str(out/'fixture')]),('whole_entry',['check_entry_CPU.py','--fixture-result',str(out/'actual-fixture.json')])]:
   log=out/(name+'.log');left=260-(time.monotonic()-start);assert left>15
   with log.open('x')as f:p=subprocess.run([sys.executable,'-B',str(d/args[0])]+args[1:],stdout=f,stderr=subprocess.STDOUT,timeout=left)
   values=[json.loads(s)for s in log.read_text().splitlines()if s.startswith('{')]
   q['phases'][name]=dict(returncode=p.returncode,log_path=str(log),log_SHA256=sha(log),result=values[-1]if values else None)
   assert p.returncode==0 and values and values[-1]['completed']and values[-1]['passed'],dict(phase=name,log=str(log))
   if name=='actual_fixture':(out/'actual-fixture.json').write_text(json.dumps(values[-1],indent=2)+'\n')
  q.update(actual_new_Native_binding_derived=q['phases']['static_binding']['result']['actual_new_Native_binding_derived'],actual_fixture_encoder_SHA256_verified=all(q['phases']['actual_fixture']['result']['fixtures'][k]['actual_prompt_tokens']==n for k,n in [('L',8489),('S',150)]),passed=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  q.update(completed=True,seconds=time.monotonic()-start);(out/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps(dict(completed=True,passed=q['passed'],GPU=0,actual_CPU=step,receipt=str(out/'CPU-result.json'),error=q.get('error'))));raise SystemExit(0 if q['passed']else 1)
if __name__=='__main__':main()
