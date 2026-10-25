import pathlib,json,os,traceback
from Kimi_A8_ABC_real_r8996 import run
r=pathlib.Path('/refactor');handle=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('Kimi-A8-ABC-real-r8996-'+handle+'.json')
try:run(r/'incoming/manifest_ABC_r8996.json',out)
except BaseException as e:
 q=json.loads(out.read_text())if out.exists()else{};q.update(completed=False,error=repr(e),traceback=traceback.format_exc());tmp=out.with_suffix('.tmp');tmp.write_bytes((json.dumps(q,indent=2)+'\n').encode());tmp.replace(out);raise
