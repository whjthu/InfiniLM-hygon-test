from pathlib import Path
import json,hashlib,subprocess,sys,time,os,traceback
D=Path(__file__).resolve().parent;O=Path('/refactor/candidates')/D.name
O.mkdir(exist_ok=False);start=time.monotonic();q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,checkpoint_loaded=False,source_manifest_SHA256=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
try:
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
 assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in m['dependencies_SHA256'].items())
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 p=subprocess.run([sys.executable,'-B',str(D/'check_CPU.py')],text=True,capture_output=True,timeout=540)
 (O/'checks.log').write_text(p.stdout+p.stderr);q['returncode']=p.returncode
 assert p.returncode==0,'Keep actual CPU failure; no GPU'
 q['checks']=json.loads([l for l in p.stdout.splitlines()if l.startswith('{')][-1]);assert q['checks']['passed']
 q['passed']=True
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
 q.update(completed=True,elapsed_s=time.monotonic()-start);(O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
raise SystemExit(0 if q['passed']else 1)
