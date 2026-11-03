"""Actual CLI preflight path never calls Slurm or dlopens the candidate Native."""
from pathlib import Path
import json,os,subprocess,tempfile,time
from contract_r410 import environment,publish_once,RUNTIME_SHA,NATIVE_SHA,NATIVE_SECONDS,sha
def run(root,directory,plan):
 root=Path(root);d=Path(directory);runtime=d.parent/'runtime-v3';assert root==Path('/refactor')
 assert all(os.environ.get(k)==''for k in['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 rows=[];shell=root/'incoming/Kimi-TP8PP4-ordinary-r410/runtime-v3/target_inside.sh'
 with tempfile.TemporaryDirectory(prefix='Kimi410-final-cli-',dir=root/'results')as tmp:
  tmp=Path(tmp);claim=tmp/'recording-only-claim.json';publish_once(claim,dict(Model_GO=False,GPU_GO=False,CPU_preflight_only=True,runtime_source_manifest_SHA256=RUNTIME_SHA,target_cards=32,draft_cards=0,Native_seconds=NATIVE_SECONDS))
  epoch=time.time();values=environment(plan,'CPU-preflight-only-no-Model-'+('0'*32),epoch,'127.0.0.1',root/'control'/('Kimi-ordinary-r410-'+plan['run']+'-cancel.json'),claim)
  values['KIMI410_PUBLISHED_MODEL_CLAIM']=str(claim)
  # The CPU consumer reads only its Settings/ABI. No claim is passed to actual main.
  for node in range(4):
   result=tmp/('node'+str(node)+'.json');env=dict(os.environ,**values,SLURM_PROCID=str(node),KIMI410_PREFLIGHT_CPU_ONLY='1',KIMI410_PREFLIGHT_CONSUMER=str(d/'final_cli_consumer.py'),KIMI410_PREFLIGHT_RESULT=str(result),BASH_ENV=str(d/'preflight_hooks.sh'))
   p=subprocess.run(['/bin/bash','--noprofile','--norc',str(shell)],env=env,text=True,capture_output=True,timeout=45)
   assert p.returncode==0 and result.is_file(),dict(node=node,rc=p.returncode,stdout=p.stdout[-1000:],stderr=p.stderr[-3000:])
   q=json.loads(result.read_bytes());assert q['passed']and q['node']==node and q['Slurm_commands']==0;rows.append(q)
 return dict(completed=True,passed=True,actual_final_CLI=True,production_shell_SHA256=sha(shell),runtime_source_manifest_SHA256=RUNTIME_SHA,Native_SHA256=NATIVE_SHA,nodes=rows,GPU=0,Slurm_commands=0,Native_dlopen=False,Runtime_created=False,Model_GO=False,CPU826_not_repeated=True)
