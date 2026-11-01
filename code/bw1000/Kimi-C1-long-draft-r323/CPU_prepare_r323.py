from pathlib import Path
import json,hashlib,os,signal,datetime,traceback
signal.alarm(205)
R=Path('/refactor');D=R/'incoming/Kimi-C1-long-draft-r323';O=R/'candidates/Kimi-C1-long-draft-CPU-r323';O.mkdir(exist_ok=False)
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'GPU':0,'target':0,'weights':0,'HTTP':0}
try:
 mf=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest();assert mf==os.environ['KIMI_LONG_DRAFT_MANIFEST_SHA']
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
 q['source_manifest_SHA256']=mf
 from check_long_context_r323 import run
 q.update(run())
except BaseException as e:q.update(error=repr(e),failure_locations=traceback.format_exc()[-4000:])
finally:
 q['completed']=True;(O/'CPU-prepared.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
assert q['passed']
