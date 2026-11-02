"""Single no-GPU CPU invocation; no compilation, Native dlopen or weights."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;S=D.parent;R=Path('/refactor');O=R/'candidates/Deep-HC-teacher-forced-logits-r397-CPU-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(S/'SOURCE_MANIFEST.json')==os.environ['DEEP397_CPU_SOURCE_MANIFEST_SHA'];m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
assert os.environ['SLURM_JOB_ID']=='910407'and not O.exists(),'No replay of immutable CPU result'
q={'completed':False,'passed':False,'actual_CPU':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_manifest_SHA256':sha(S/'SOURCE_MANIFEST.json'),'owner_manifest_SHA256':sha(S/'owner/SOURCE_MANIFEST.json'),'publication_commit':os.environ['DEEP397_PUBLICATION_COMMIT'],'GPU':False,'Native_dlopen':False,'weights_opened':False,'new_compile':False,'GPU_GO':False}
began=time.time()
try:
    p=subprocess.run([sys.executable,'-B',str(D/'CPU_entry.py')],capture_output=True,text=True,timeout=230);q.update(entry_returncode=p.returncode,stdout_tail=p.stdout[-2000:],stderr_tail=p.stderr[-2000:]);assert p.returncode==0
    cp=O/'CPU-entry.json';entry=json.loads(cp.read_bytes());assert entry['completed']and entry['passed']and not entry['GPU']and not entry['Native_dlopen']and not entry['weights_opened']
    q.update(passed=True,entry_path=str(cp),entry_SHA256=sha(cp),cases=entry['cases'],actual_CPU_prompt=entry['actual_CPU_prompt'])
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
q.update(completed=True,start_unix=began,deadline_unix=began+270,seconds=time.time()-began);O.mkdir(exist_ok=True);(O/'CPU-whole.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','seconds','error']}));assert q['passed']
