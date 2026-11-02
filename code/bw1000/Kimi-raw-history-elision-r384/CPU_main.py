from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback

D=Path(__file__).resolve().parent;R=Path('/refactor')
out=R/'candidates'/'Kimi-raw-history-elision-r384-CPU'
out.mkdir(exist_ok=False)
q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,Model_loads=0,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());q['source_manifest_SHA256']=sha(D/'SOURCE_MANIFEST.json')
 assert all(sha(D/n)==h for n,h in mf['files_SHA256'].items())
 assert all(sha(Path(p))==h for p,h in mf['dependencies_SHA256'].items())
 tick=time.monotonic();p=subprocess.run([sys.executable,'-B',str(D/'check_Torch_CPU.py'),'--incoming',str(R/'incoming'),'--output',str(out/'Torch-result.json')],timeout=250)
 q['consumer_wall_s']=time.monotonic()-tick;assert p.returncode==0
 result=json.loads((out/'Torch-result.json').read_bytes());assert result['completed']and result['passed']and result['GPU']==0
 q.update(passed=True,consumer=dict(path=str(out/'Torch-result.json'),SHA256=sha(out/'Torch-result.json')),result=result)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
 q['completed']=True;(out/'CPU-result.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(q,ensure_ascii=False),flush=True)
if not q['passed']:raise SystemExit(1)
