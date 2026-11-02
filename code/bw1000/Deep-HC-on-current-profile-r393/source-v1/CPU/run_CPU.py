"""One CPU whole-entry invocation; never load Native/Torch/weights or compile."""
from pathlib import Path
import json,hashlib,subprocess,sys,os,time,traceback
D=Path(__file__).resolve().parent;S=D.parent;R=Path('/refactor');O=R/'candidates/Deep-HC-on-current-profile-r393-CPU-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mf=S/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['DEEP393_CPU_SOURCE_MANIFEST_SHA'];m=json.loads(mf.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
assert not O.exists(),'Immutable CPU result already exists; no replay';assert os.environ['SLURM_JOB_ID']=='910407'
q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(mf),owner_manifest_SHA256=sha(S/'owner/SOURCE_MANIFEST.json'),publication_commit=os.environ['DEEP393_PUBLICATION_COMMIT'],GPU=False,Slurm_GPU=False,Native_dlopen=False,weights_opened=False,new_compile=False,GPU_GO=False)
began=time.monotonic()
try:
 p=subprocess.run([sys.executable,'-B',str(D/'check_contract_CPU.py')],text=True,capture_output=True,timeout=40);assert p.returncode==0,p.stderr;q['local_wrapper_contract']=json.loads(p.stdout);assert q['local_wrapper_contract']['passed']
 p=subprocess.run([sys.executable,'-B',str(D/'CPU_entry.py'),'--owner-sha',q['owner_manifest_SHA256']],text=True,capture_output=True,timeout=200);q.update(entry_returncode=p.returncode,entry_stdout_tail=p.stdout[-3000:],entry_stderr_tail=p.stderr[-3000:]);assert p.returncode==0
 cp=O/'CPU-entry.json';entry=json.loads(cp.read_bytes());assert entry['completed']and entry['passed']and not entry['GPU']and not entry['Native_dlopen']and not entry['weights_opened'];q.update(entry_path=str(cp),entry_SHA256=sha(cp),entry_case_count=len(entry['cases']),cases=entry['cases'],passed=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q.update(completed=True,seconds=time.monotonic()-began);O.mkdir(exist_ok=True);p=O/'CPU-whole.json';assert not p.exists();p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','seconds','error']}));assert q['passed']
