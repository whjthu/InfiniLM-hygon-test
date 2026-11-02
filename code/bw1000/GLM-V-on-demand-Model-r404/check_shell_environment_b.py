"""Actual Bash consumption of the production shell policy fragment, without GPU.

This executes the original role export and every required-policy assertion plus
network exports. DTK/import/Native/weights execution is explicitly outside scope.
"""
from pathlib import Path
import ast,hashlib,json,os,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(source):
 source=Path(source);inside=source/'owner/GPU_inside_r290.sh';raw=inside.read_text();controller=source/'controller/controller_v4.py'
 tree=ast.parse(controller.read_text());updates=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='update'and any(k.arg=='WAVE_INSIDE_R2323'for k in n.keywords)];assert len(updates)==1
 env={k.arg:str(ast.literal_eval(k.value))for k in updates[0].keywords if k.arg and isinstance(k.value,ast.Constant)}
 assert env['GLM_BOUNDED_ROPE_R4279']==env['GLM_SHARED_CONTROL_R4279']=='1'
 role=next(line for line in raw.splitlines()if line.startswith('export GLM_STAGE_R5040='))
 fragment=raw[raw.index('test "${GLM_EXACT_BANKS_R4188:?}"'):raw.index('python3 /refactor/incoming/GLM_transient_preflight_r9059.py')]
 shell='set -euo pipefail\n'+role+'\n'+fragment+'\ntest "$GLM_STAGE_R5040" = "$EXPECTED_STAGE"\ntest "$GLM_BOUNDED_ROPE_R4279" = 1\ntest "$GLM_SHARED_CONTROL_R4279" = 1\nprintf "ENV_ROLE_CONSUMED stage=%s PP=%s rows=%s capacity=%s\\n" "$GLM_STAGE_R5040" "$GLM_PP_SIZE_R3603" "$GLM_CACHE_ROWS_R3627" "$GLM_CACHE_CAP_R3627"\n'
 results=[]
 for stage in [0,1]:
  base=dict(PATH=os.environ.get('PATH','/usr/bin:/bin'),**env,SLURM_PROCID=str(stage),EXPECTED_STAGE=str(stage))
  for missing in [None,'GLM_BOUNDED_ROPE_R4279','GLM_SHARED_CONTROL_R4279']:
   e=dict(base)
   if missing:e.pop(missing)
   p=subprocess.run(['bash','--noprofile','--norc','-c',shell],env=e,text=True,capture_output=True,timeout=10)
   results.append(dict(stage=stage,removed=missing,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=(p.returncode==0 if missing is None else p.returncode!=0 and missing in p.stderr)))
   assert results[-1]['passed'],results[-1]
 return dict(completed=True,passed=True,actual_Bash_execution=True,source_policy_fragment_SHA256=hashlib.sha256(shell.encode()).hexdigest(),GPU_inside_SHA256=sha(inside),controller_SHA256=sha(controller),cases=results,source_scope='Original production role export and complete required policy/network fragment; no DTK/Torch/Native/checkpoint/device execution',Slurm=0,GPU=0,Native_dlopen=False,Model=0)
if __name__=='__main__':print(json.dumps(check(Path(__file__).resolve().parent)))
