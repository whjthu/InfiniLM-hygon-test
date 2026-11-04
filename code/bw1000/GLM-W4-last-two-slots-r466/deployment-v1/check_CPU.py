"""Necessary changed deployment metadata/actual nonserve wholeCLI only."""
from pathlib import Path
import contextlib,datetime,hashlib,io,json,os,subprocess,time,traceback
D=Path(__file__).resolve().parent;R=D.parents[2];start=time.monotonic()
out=R/'candidates/GLM466-last-two-deployment-CPU-v1/result.json'
q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,SDK_execution=False,weights=False,new_compiles=0,encoding_repeated=False,slots=[4,5],actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
try:
 import controller as C
 digest=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest();q['source_manifest_SHA256']=digest
 proofs=[]
 for slot in [4,5]:
  capture=io.StringIO()
  with contextlib.redirect_stdout(capture):rc=C.main(['--manifest-sha',digest,'--publication-commit',os.environ['GLM466_SOURCE_COMMIT'],'--preflight-only','--slot',str(slot)])
  proof=json.loads(capture.getvalue());assert rc==0 and proof['passed']and proof['GPU']==proof['Slurm']==0
  assert proof['proof']['slot']==slot and proof['proof']['questions']==32 and proof['proof']['HTTP']==64
  proofs.append(proof)
  assert C.SLOT['nodes']==(['a02r4n14','a02r4n15']if slot==4 else ['a02r4n12','a02r4n13'])
 # No launch/intent/Popen/device boundary is entered by C.main --preflight-only.
 for rejected in [0,1,2,3,6,7,-1,8]:
  try:C.configure_slot(rejected)
  except AssertionError:pass
  else:raise AssertionError('Unclaimed slot admitted')
 maps=Path('/proc/self/maps').read_text();assert not any(s in maps for s in ['libInfiniRT','libinfiniop','libInfiniCCL','libGLM_V_on_demand'])
 q.update(proofs=proofs,unclaimed_slots_rejected=8,passed=True,actual_public_nonserve_CLI_roles=4,project_or_SDK_maps=[],no_intent_or_job_created=True)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q.update(completed=True,wall_s=time.monotonic()-start,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x')as f:json.dump(q,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=q['passed'],actual_CPU=q['actual_CPU'],receipt=str(out),wall_s=q['wall_s'])))
if not q['passed']:raise SystemExit(1)
