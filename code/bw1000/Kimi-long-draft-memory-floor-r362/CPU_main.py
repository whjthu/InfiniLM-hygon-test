from pathlib import Path
import hashlib,json,os,subprocess,sys,time

D=Path(__file__).resolve().parent;O=Path('/refactor/candidates')/D.name
O.mkdir(exist_ok=False)
start=time.monotonic();q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,
    source_manifest_SHA256=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
    actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
try:
    assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
    m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
    assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
    p=subprocess.run([sys.executable,'-B',str(D/'check_CPU.py')],text=True,capture_output=True,timeout=240)
    log=O/'checks.log';log.write_text(p.stdout+p.stderr)
    q.update(returncode=p.returncode,log_SHA256=hashlib.sha256(log.read_bytes()).hexdigest())
    assert p.returncode==0,'Private worker CPU contract failed; preserve log'
    data=json.loads([x for x in p.stdout.splitlines()if x.startswith('{')][-1])
    assert data['passed']and data['GPU']==0
    q.update(passed=True,checks=data)
except BaseException as error:q.update(error=repr(error))
finally:
    q.update(completed=True,elapsed_s=time.monotonic()-start)
    (O/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(q))
raise SystemExit(0 if q['passed']else 1)
