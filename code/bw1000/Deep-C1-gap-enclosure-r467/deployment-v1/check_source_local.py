"""Actual source-only producer/worker/keeper and pending-CPU refusal; no Slurm."""
from pathlib import Path
from unittest.mock import patch
import contextlib,io,json,os,sys
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import support,launch_once,controller,worker

def check():
 digest=support.sha(D/'SOURCE_MANIFEST.json');calls=[]
 def forbidden(*a,**k):calls.append(a);raise AssertionError('No subprocess/Slurm permitted in source-only consumer')
 streams=[]
 with patch('subprocess.Popen',forbidden),patch('subprocess.check_output',forbidden),patch('subprocess.run',forbidden):
  for module in[launch_once,controller]:
   with contextlib.redirect_stdout(io.StringIO())as out:assert module.main(['--manifest-sha',digest,'--publication-commit','0'*40,'--source-only'])==0
   q=json.loads(out.getvalue());assert q['source_verified']and q['execution_READY']is False and q['GPU']==0 and q['Slurm']==0;streams.append(q)
  with patch.dict(os.environ,{'DEEP467_PROFILE_DEPLOY_SHA':digest}),contextlib.redirect_stdout(io.StringIO())as out:assert worker.main(['--source-only'])==0
  q=json.loads(out.getvalue());assert q['source_verified']and not q['execution_READY']and q['GPU']is False and not q['Native_dlopen'];streams.append(q)
  rejected=False
  try:launch_once.main(['--manifest-sha',digest,'--publication-commit','0'*40])
  except AssertionError as e:rejected='CPU receipt' in str(e)
  assert rejected and not calls,'UnsubmittedCPU must stop before any subprocess'
 gpu=controller.gpu_command();keeper=launch_once.keeper_command(digest,'0'*40)
 assert '--jobid=910374'in gpu and gpu[gpu.index('-w')+1]=='a02r4n04'and '--gres=dcu:8'in gpu and gpu[gpu.index('-t')+1]=='00:45:00'
 assert '--jobid=910407'in keeper and keeper[keeper.index('-w')+1]=='a14r2n20'and '--gres=none'in keeper and keeper[keeper.index('-t')+1]=='00:50:00'
 cfg=json.loads((D/'config.json').read_bytes());support.validate_config(cfg);assert cfg['gpu_start_epoch']is None and not cfg['execute_authorized']
 cleaned=support.clean_environment(dict(BASH_ENV='x',PYTHONSTARTUP='x',SLURM_JOB_ID='wrong',DEEP_X='wrong',INFINILM_DEEP_X='wrong',SLURM_CONF='keep',PATH='keep'))
 assert cleaned==dict(SLURM_CONF='keep',PATH='keep')
 assert not any(n in sys.modules for n in['torch','_infinilm','flash_attn'])
 return dict(passed=True,actual_source_only_producer_keeper_worker=True,pending_real_CPU_blocks_launch_before_Popen=True,actual_gpu_keeper_argv_scope=True,source_schema_and_filehandoff=True,subprocess_calls=0,Slurm=0,Native_dlopen=False,SDK_Runtime=False,HTTP=0,GPU=False,Model_READY=False,source_manifest_under_test=digest)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();q=check();Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
