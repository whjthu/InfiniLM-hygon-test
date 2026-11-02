"""One bounded real static binding/fixture and shared wholeentry CPU qualification."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,traceback

def main():
 d=Path(__file__).resolve().parent;root=Path('/refactor');step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=root/'candidates'/('Deep-PD-execution-r382-CPU-'+step);out.mkdir(exist_ok=False)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,weights_opened=False,actual_CPU=step,phases={});start=time.monotonic()
 try:
  mf=d/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP_PD_CPU_SOURCE_SHA'];source=json.loads(mf.read_bytes());assert source['GPU_GO']is False and all(sha(d/n)==h for n,h in source['files_SHA256'].items());q['source_manifest_SHA256']=sha(mf)
  for name,args in [('static_binding',['CPU_static.py']),('actual_fixture',['CPU_fixture.py',str(out/'fixture')]),('whole_entry',['check_entry_CPU.py'])]:
   log=out/(name+'.log');left=540-(time.monotonic()-start);assert left>30
   with log.open('x')as handle:p=subprocess.run([sys.executable,'-B',str(d/args[0])]+args[1:],stdout=handle,stderr=subprocess.STDOUT,timeout=left)
   text=log.read_text();values=[json.loads(s)for s in text.splitlines()if s.startswith('{')];q['phases'][name]=dict(returncode=p.returncode,log_path=str(log),log_SHA256=sha(log),result=values[-1]if values else None)
   assert p.returncode==0 and values and values[-1]['completed']and values[-1]['passed'],dict(phase=name,log=str(log))
  q.update(actual_new_Native_binding_derived=q['phases']['static_binding']['result']['actual_new_Native_binding_derived'],actual_fixture_encoder_SHA256_verified=q['phases']['actual_fixture']['result']['prompt_IDs_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df',passed=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,seconds=time.monotonic()-start);(out/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps(dict(completed=True,passed=q['passed'],GPU=0,actual_CPU=step,receipt=str(out/'CPU-result.json'),error=q.get('error'))));raise SystemExit(0 if q['passed']else 1)
if __name__=='__main__':main()
